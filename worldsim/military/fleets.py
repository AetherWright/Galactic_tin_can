"""Space fleet model, ship templates, space combat and division transport.

Design goals
------------
* **Modular ship types** — add new hull classes to :data:`SHIP_TEMPLATES`.
* **Simple combat** — :func:`resolve_space_combat` isolates the formula so it
  can be upgraded without touching anything else.
* **Division transport** — fleets carry ground divisions between planets/stars;
  capacity is the sum of ``ShipTemplate.capacity`` across all ships.
* **FTL integration** — travel speed is driven by the :mod:`ftl` registry via
  :func:`~worldsim.military.ftl.best_ftl_drive`.

Glossary
--------
location:
    The planet or star name where the fleet currently sits.
mission:
    ``"idle"`` | ``"moving"`` | ``"combat"`` — the fleet's current activity.
travel_turns_remaining:
    Integer countdown in simulation *fifths* until arrival.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..core import vprint, wprint

if TYPE_CHECKING:
    from ..nations.nation import Nation
    from .divisions import Division


# ---------------------------------------------------------------------------
# Ship templates
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ShipTemplate:
    """Blueprint describing a ship hull class.

    Parameters
    ----------
    name:
        Unique hull identifier — must match the key in :data:`SHIP_TEMPLATES`.
    firepower:
        Damage output *per ship* applied each combat round.
    hp:
        Hull-points *per ship*.  The fleet loses ships when cumulative damage
        reaches their hp threshold.
    capacity:
        Maximum soldiers this hull can carry.  ``0`` = pure combat ship.
    build_cost:
        Resource requirements to produce **one** ship of this class.
    """

    name: str
    firepower: float = 10.0
    hp: float = 20.0
    capacity: int = 0
    build_cost: Dict[str, float] = field(default_factory=dict)


# Registry — add new hull types here only
SHIP_TEMPLATES: Dict[str, ShipTemplate] = {
    "Frigate": ShipTemplate(
        "Frigate",
        firepower=15.0,
        hp=10.0,
        capacity=0,
        build_cost={"metal": 30.0, "energy": 10.0},
    ),
    "Cruiser": ShipTemplate(
        "Cruiser",
        firepower=40.0,
        hp=30.0,
        capacity=0,
        build_cost={"metal": 80.0, "energy": 25.0},
    ),
    "Transport": ShipTemplate(
        "Transport",
        firepower=2.0,
        hp=15.0,
        capacity=2000,      # carries up to 2 000 soldiers per ship
        build_cost={"metal": 50.0, "energy": 15.0},
    ),
    "Battleship": ShipTemplate(
        "Battleship",
        firepower=100.0,
        hp=80.0,
        capacity=0,
        build_cost={"metal": 200.0, "energy": 60.0},
    ),
}

# Stable name list for AI indexing
SHIP_TEMPLATE_NAMES: List[str] = list(SHIP_TEMPLATES.keys())


# ---------------------------------------------------------------------------
# Fleet
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Fleet:
    """A space fleet owned by a nation.

    Parameters
    ----------
    fleet_id:
        Unique index within the owning nation's fleet list.
    nation_id:
        Owner nation identifier.
    location:
        Current planet or star name.
    ships:
        Mapping of :attr:`ShipTemplate.name` → ship count.
    divisions:
        Ground divisions currently embarked on this fleet.
    mission:
        Current task: ``"idle"``, ``"moving"``, or ``"combat"``.
    destination:
        Target location name when ``mission == "moving"``.
    travel_turns_remaining:
        Simulation fifths until the fleet arrives at ``destination``.
    ftl_cooldown:
        Fifths remaining before another FTL jump is permitted.
    """

    fleet_id: int
    nation_id: int
    location: str
    ships: Dict[str, int] = field(default_factory=dict)
    divisions: List["Division"] = field(default_factory=list)
    mission: str = "idle"
    destination: Optional[str] = None
    travel_turns_remaining: int = 0
    ftl_cooldown: int = 0
    # FSM state and neural controller (set lazily on first use)
    state: str = "patrol"
    controller: Any = field(default=None)

    def __post_init__(self) -> None:
        """Lazily attach a fleet controller if none was supplied.

        The import is deferred to avoid a circular dependency between
        ``fleet.py`` and ``military_ai.py`` at module load time.
        """
        if self.controller is None:
            try:
                from .command import make_fleet_controller
                self.controller = make_fleet_controller()
            except Exception:  # pragma: no cover — only fails in extreme edge cases
                pass

    # ------------------------------------------------------------------
    # Derived stats
    # ------------------------------------------------------------------

    @property
    def total_firepower(self) -> float:
        """Combined weapon output of all ships."""
        total = 0.0
        for name, count in self.ships.items():
            tmpl = SHIP_TEMPLATES.get(name)
            if tmpl:
                total += tmpl.firepower * count
        return total

    @property
    def total_hp(self) -> float:
        """Combined hull points of all ships."""
        total = 0.0
        for name, count in self.ships.items():
            tmpl = SHIP_TEMPLATES.get(name)
            if tmpl:
                total += tmpl.hp * count
        return total

    @property
    def troop_capacity(self) -> int:
        """Maximum soldiers that can be embarked."""
        total = 0
        for name, count in self.ships.items():
            tmpl = SHIP_TEMPLATES.get(name)
            if tmpl:
                total += tmpl.capacity * count
        return total

    @property
    def total_ships(self) -> int:
        return sum(self.ships.values())

    def is_empty(self) -> bool:
        """Return ``True`` if the fleet has no ships left."""
        return self.total_ships == 0

    # ------------------------------------------------------------------
    # Ship management
    # ------------------------------------------------------------------

    def add_ships(self, template_name: str, count: int = 1) -> None:
        """Add ``count`` ships of ``template_name`` to this fleet."""
        if template_name not in SHIP_TEMPLATES:
            return
        self.ships[template_name] = self.ships.get(template_name, 0) + count

    def remove_ships(self, template_name: str, count: int = 1) -> None:
        """Remove up to ``count`` ships of ``template_name``."""
        current = self.ships.get(template_name, 0)
        remaining = max(0, current - count)
        if remaining == 0:
            self.ships.pop(template_name, None)
        else:
            self.ships[template_name] = remaining

    # ------------------------------------------------------------------
    # Division transport
    # ------------------------------------------------------------------

    def load_divisions(self, divisions: List["Division"]) -> List["Division"]:
        """Embark as many ``divisions`` as capacity allows.

        Returns the list of divisions that could **not** be loaded because the
        fleet was at capacity.
        """
        overflow: List["Division"] = []
        for div in divisions:
            used = sum(d.soldiers for d in self.divisions)
            if used + div.soldiers <= self.troop_capacity:
                self.divisions.append(div)
            else:
                overflow.append(div)
        return overflow

    def unload_divisions(self) -> List["Division"]:
        """Disembark all embarked divisions and return them."""
        unloaded = list(self.divisions)
        self.divisions.clear()
        return unloaded

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def begin_move(
        self,
        destination: str,
        distance_ly: float,
        *,
        ftl_speed_mult: float = 1.0,
        cooldown: int = 0,
    ) -> None:
        """Set the fleet on a course toward ``destination``.

        Parameters
        ----------
        destination:
            Planet or star name to travel to.
        distance_ly:
            Distance in light-years (star coordinate units).
        ftl_speed_mult:
            Speed divisor from the nation's best FTL drive.
        cooldown:
            FTL cooldown turns to impose after departure.
        """
        import math

        self.mission = "moving"
        self.destination = destination
        raw = distance_ly / max(0.01, ftl_speed_mult)
        self.travel_turns_remaining = max(1, math.ceil(raw))
        self.ftl_cooldown = cooldown

    def advance_movement(self) -> bool:
        """Tick the travel counter by one fifth.

        Returns ``True`` when the fleet has arrived at its destination.
        """
        if self.mission != "moving":
            return False
        self.travel_turns_remaining = max(0, self.travel_turns_remaining - 1)
        if self.travel_turns_remaining <= 0:
            self.location = self.destination or self.location
            self.destination = None
            self.mission = "idle"
            return True
        return False

    def tick_cooldown(self) -> None:
        """Decrement FTL cooldown counter by one fifth."""
        if self.ftl_cooldown > 0:
            self.ftl_cooldown -= 1


# ---------------------------------------------------------------------------
# Space combat
# ---------------------------------------------------------------------------

def resolve_space_combat(
    attacker: Fleet,
    defender: Fleet,
    *,
    attacker_nation: Optional["Nation"] = None,
    defender_nation: Optional["Nation"] = None,
    max_rounds: int = 5,
) -> str:
    """Fight a space battle for up to ``max_rounds``.

    Each round, both sides deal randomised firepower (±20 %) to the other's
    total HP pool.  After all rounds the proportional survival rate is applied
    back to each hull class.

    Parameters
    ----------
    attacker, defender:
        The two fleets engaged in combat.
    attacker_nation, defender_nation:
        Optional owning nations used to apply ``tech_bonuses["division_power"]``
        as a combat modifier (same key as ground units for simplicity).
    max_rounds:
        Hard ceiling on combat rounds.  Increase for longer, bloodier battles.

    Returns
    -------
    ``"attacker"`` if the attacker's fleet survives and the defender's does not,
    ``"defender"`` if vice-versa, or ``"draw"`` if both (or neither) survive.
    """
    if attacker.is_empty() or defender.is_empty():
        return "draw"

    tech_atk = 1.0
    tech_def = 1.0
    if attacker_nation:
        tech_atk = attacker_nation.tech_bonuses.get("division_power", 1.0)
    if defender_nation:
        tech_def = defender_nation.tech_bonuses.get("division_power", 1.0)

    atk_hp = attacker.total_hp
    def_hp = defender.total_hp

    for _ in range(max_rounds):
        if atk_hp <= 0 or def_hp <= 0:
            break
        # Attacker deals damage to defender's HP pool and vice-versa
        def_lost = attacker.total_firepower * tech_atk * random.uniform(0.8, 1.2)
        atk_lost = defender.total_firepower * tech_def * random.uniform(0.8, 1.2)
        atk_hp = max(0.0, atk_hp - atk_lost)
        def_hp = max(0.0, def_hp - def_lost)

    # Apply proportional ship losses back to each hull class
    total_atk_hp = attacker.total_hp
    if total_atk_hp > 0:
        survival = max(0.0, atk_hp / total_atk_hp)
        for name in list(attacker.ships):
            count = attacker.ships[name]
            attacker.ships[name] = max(0, int(count * survival))
            if attacker.ships[name] == 0:
                del attacker.ships[name]

    total_def_hp = defender.total_hp
    if total_def_hp > 0:
        survival = max(0.0, def_hp / total_def_hp)
        for name in list(defender.ships):
            count = defender.ships[name]
            defender.ships[name] = max(0, int(count * survival))
            if defender.ships[name] == 0:
                del defender.ships[name]

    atk_alive = not attacker.is_empty()
    def_alive = not defender.is_empty()

    if atk_alive and not def_alive:
        wprint(
            str(attacker.nation_id),
            f"  Fleet#{attacker.fleet_id} destroyed Fleet#{defender.fleet_id}"
        )
        return "attacker"
    if def_alive and not atk_alive:
        wprint(
            str(defender.nation_id),
            f"  Fleet#{defender.fleet_id} repelled Fleet#{attacker.fleet_id}"
        )
        return "defender"
    return "draw"


# ---------------------------------------------------------------------------
# Nation-level fleet helpers
# ---------------------------------------------------------------------------

def build_fleet_ship(nation: "Nation", template_name: str = "Frigate") -> bool:
    """Attempt to build one ship of ``template_name`` for ``nation``.

    Requirements:
    * At least one spaceport or shipyard.
    * Sufficient resources for the ship's :attr:`ShipTemplate.build_cost`.

    The new ship is added to an existing idle fleet at the nation's home
    planet, or a new fleet is created.

    Returns ``True`` if the ship was built.
    """
    if not nation.spaceports and not nation.shipyards:
        return False
    tmpl = SHIP_TEMPLATES.get(template_name)
    if not tmpl:
        return False
    if not nation.has_resources(tmpl.build_cost):
        return False
    nation.spend_resources(tmpl.build_cost)

    # Find an existing idle fleet at home planet or create one
    target: Optional[Fleet] = None
    for fl in nation.fleets:
        if fl.location == nation.planet and fl.mission == "idle":
            target = fl
            break
    if target is None:
        fleet_id = len(nation.fleets)
        target = Fleet(fleet_id, nation.id, nation.planet)
        nation.fleets.append(target)

    target.add_ships(template_name)
    return True


def process_fleet_movement(
    nation: "Nation",
    nations: Dict[int, "Nation"],  # type: ignore[name-defined]
) -> None:
    """Advance all of ``nation``'s fleets one simulation fifth.

    For each fleet:

    1. Tick the FTL cooldown counter.
    2. Advance movement; if the fleet arrives, unload embarked divisions
       back into the nation's division pool.
    3. If hostile fleets are present at the arrival location and the two
       nations are at war, trigger :func:`resolve_space_combat`.
    4. Purge empty fleets.
    """
    for fleet in list(nation.fleets):
        fleet.tick_cooldown()

        arrived = fleet.advance_movement()
        if not arrived:
            continue

        # Unload troops at the destination planet/star
        if fleet.divisions:
            unloaded = fleet.unload_divisions()
            nation.divisions.extend(unloaded)
            wprint(
                nation.name,
                f"  {nation.name} Fleet#{fleet.fleet_id} landed {len(unloaded)}div @ {fleet.location}"
            )

        # Check for enemy fleets sharing the same location
        for other in list(nations.values()):
            if other.id == nation.id:
                continue
            if other.id not in nation.at_war:
                continue
            for enemy_fleet in list(other.fleets):
                if enemy_fleet.location != fleet.location:
                    continue
                if fleet.is_empty() and enemy_fleet.is_empty():
                    continue
                result = resolve_space_combat(
                    fleet,
                    enemy_fleet,
                    attacker_nation=nation,
                    defender_nation=other,
                )
                if result == "attacker" and enemy_fleet.is_empty():
                    other.fleets.remove(enemy_fleet)
                elif result == "defender" and fleet.is_empty():
                    break  # our fleet was destroyed; stop checking

    # Remove any fleets that were wiped out in combat
    nation.fleets = [f for f in nation.fleets if not f.is_empty()]


def order_fleet_move(
    nation: "Nation",
    fleet: Fleet,
    destination: str,
    distance_ly: float,
    divisions_to_load: Optional[List["Division"]] = None,
) -> bool:
    """Order ``fleet`` to travel to ``destination``, optionally loading troops.

    The order is rejected if the fleet is already underway or if the FTL drive
    is on cooldown.

    Parameters
    ----------
    nation:
        Owning nation (used to look up the best FTL drive).
    fleet:
        The fleet to order.
    destination:
        Name of the target planet or star.
    distance_ly:
        Distance to ``destination`` in light-years.
    divisions_to_load:
        Optional list of divisions to embark before departure.  Any that
        exceed capacity are left behind.

    Returns ``True`` if the order was accepted.
    """
    from .ftl import best_ftl_drive

    if fleet.mission == "moving":
        return False
    if fleet.ftl_cooldown > 0:
        return False

    drive = best_ftl_drive(nation)

    # Range check (0.0 = unlimited)
    if drive.range_ly > 0.0 and distance_ly > drive.range_ly:
        return False

    if divisions_to_load:
        fleet.load_divisions(divisions_to_load)

    fleet.begin_move(
        destination,
        distance_ly,
        ftl_speed_mult=drive.speed_mult,
        cooldown=drive.cooldown_turns,
    )
    return True
