from __future__ import annotations
import math
from dataclasses import dataclass, field
import random
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from ..planets import PLANETS
from ..utils import distance, vprint, wprint, logistic_growth
from ..ai import SimplePerceptron
from ..representations import build_alliance_matrix, galactic_control_layers

if TYPE_CHECKING:
    from ..models.nation import Nation
    from ..planets.planet import Planet


# ---------------------------------------------------------------------------
# Doctrine → maximum manpower fraction of total population
# ---------------------------------------------------------------------------

_DOCTRINE_MANPOWER: Dict[str, float] = {
    "total_war":         0.05,   # extreme wartime mobilization
    "offensive":         0.008,  
    "defensive":         0.007,
    "strategic_reserve": 0.015,  # peacetime standing force
    "economic":          0.002,
}
_DEFAULT_MANPOWER: float = 0.08


ORDER_TO_LABEL = {
    "attack": [1.0, 0.0, 0.0],
    "defend": [0.0, 1.0, 0.0],
    "reserve": [0.0, 0.0, 1.0],
}

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

@dataclass(slots=True)
class DivisionTemplate:
    """Template for creating specialised divisions."""

    name: str
    base_size: int
    equipment_mod: float = 1.0
    experience_mod: float = 1.0


@dataclass(slots=True)
class Division:
    soldiers: int
    x: int
    y: int
    planet: str
    experience: float = 1.0
    equipment: float = 1.0
    template: str = "Infantry"
    doctrine: str = "Balanced"
    controller: SimplePerceptron = field(
        default_factory=lambda: SimplePerceptron(3, n_outputs=3)
    )
    movement_controller: SimplePerceptron = field(
        default_factory=lambda: SimplePerceptron(6, n_outputs=3)
    )
    posture: str = "reserve"
    order: str = "reserve"
    in_transit: bool = False

    def decide_posture(self, context=None) -> str:
        if context is None:
            context = [self.soldiers / 1000.0, self.experience, self.equipment]
        scores = self.controller.predict_prob(context)
        idx = scores.index(max(scores)) if isinstance(scores, list) else 0
        self.posture = ["attack", "defend", "reserve"][idx]
        return self.posture

    def decide_movement(self, context: List[float]) -> str:
        """Return 'stay', 'deploy', or 'recall'."""
        scores = self.movement_controller.predict_prob(context)
        idx = scores.index(max(scores)) if isinstance(scores, list) else 0
        return ["stay", "deploy", "recall"][idx]

    @property
    def power(self) -> float:
        return self.soldiers * self.experience * self.equipment

def move_division(
    div: Division,
    nation: "Nation",
    enemy: "Nation",
    nations: Dict[int, "Nation"],
) -> None:

    dist_enemy = min(
        ((div.x - c.x)**2 + (div.y - c.y)**2)**0.5
        for c in enemy.cities
    ) if enemy.cities else 999.0
    dist_spaceport = min(
        ((div.x - s.x)**2 + (div.y - s.y)**2)**0.5
        for s in nation.spaceports
    ) if nation.spaceports else 999.0
    on_enemy_planet = 1.0 if div.planet == enemy.planet else 0.0
    supply = _supply_throughput(nation, PLANETS.get(div.planet))
    pop_ratio = nation.population / max(1, enemy.population)
    context = [
        dist_enemy / 1000.0,
        dist_spaceport / 1000.0,
        supply,
        on_enemy_planet,
        float(div.order == "attack"),
        min(2.0, pop_ratio),
    ]
    decision = div.decide_movement(context)
    
    if decision == "deploy" and div.planet == nation.planet:
        # Find friendly spaceports on enemy planet first
        friendly_ports = [
            s for nid in nation.alliances
            if (ally := nations.get(nid))
            for s in ally.spaceports
            if s.planet == enemy.planet
        ]
        # Fall back to own spaceports on enemy planet (previous captures)
        friendly_ports += [s for s in nation.spaceports if s.planet == enemy.planet]
        
        if friendly_ports:
            target = random.choice(friendly_ports)
            div.planet = enemy.planet
            div.x = target.x
            div.y = target.y
        elif enemy.spaceports:
            # Deep strike — riskier, costs experience
            target = random.choice(enemy.spaceports)
            div.planet = enemy.planet
            div.x = target.x
            div.y = target.y
            div.experience = max(0.1, div.experience * 0.85)
            div.equipment = max(0.1, div.equipment * 0.80)

    elif decision == "recall" and div.planet != nation.planet:
        home_ports = [s for s in nation.spaceports if s.planet == nation.planet]
        if home_ports:
            target = random.choice(home_ports)
            div.planet = nation.planet
            div.x = target.x
            div.y = target.y

@dataclass(slots=True)
class AllianceBloc:
    """Group of allied nations treated as a single entity."""

    id: int
    members: Set[int]
    centroid: Tuple[float, float, float]
    military: float

    @classmethod
    def from_members(
        cls, bloc_id: int, members: Set[int], nations: Dict[int, "Nation"]
    ) -> "AllianceBloc":
        x = y = z = 0.0
        m = 0.0
        for nid in members:
            n = nations[nid]
            wx, wy, wz = n.world_centroid
            x += wx
            y += wy
            z += wz
            m += n.military
        if members:
            c = (x / len(members), y / len(members), z / len(members))
        else:
            c = (0.0, 0.0, 0.0)
        return cls(bloc_id, members, c, m)


def issue_orders(
    nation: "Nation", alliance_row: List[int] | None = None
) -> None:
    """Assign attack/defend/reserve orders to each division.

    Reads ``nation.doctrine_signal`` (set by :class:`DoctrineAI` at the end
    of the previous fifth) and applies doctrine-level overrides on top of the
    ``WarAI``'s tactical recommendation:

    * ``total_war``        → all divisions attack regardless of WarAI
    * ``strategic_reserve`` → all divisions stand down (reserve)
    * ``defensive``        → WarAI decides, but attack is overridden to defend
    * ``offensive``        → WarAI decides, but defend is flipped to attack
    * ``economic``         → WarAI decides unconstrained

    The spatial grid and alliance row inputs are unchanged from the original
    implementation so the WarAI's weights remain compatible.
    """

    if not nation.military_ai:
        return

    # Doctrine signal written by DoctrineAI (end of previous process_turn).
    # Use getattr so nations without the field still work.
    doctrine = getattr(nation, "doctrine_signal", "defensive")

    grid_size = getattr(nation.military_ai, "grid_size", 8)
    friendly_grid, hostile_grid = galactic_control_layers(
        nation, grid_size=grid_size
    )
    grid_features = [
        val for row in friendly_grid for val in row
    ] + [val for row in hostile_grid for val in row]

    allies_dim = getattr(nation.military_ai, "allies_dim", 1)
    ally_features: List[float]
    if alliance_row is None:
        ally_features = [0.0] * allies_dim
    else:
        row = list(alliance_row)
        nation.military_ai.set_allies_dimension(len(row))
        allies_dim = nation.military_ai.allies_dim
        row = row[:allies_dim]
        if len(row) < allies_dim:
            row.extend([0.0] * (allies_dim - len(row)))
        ally_features = [float(v) for v in row]

    for div in nation.divisions:
        # Hard doctrine overrides — bypass WarAI entirely
        if doctrine == "total_war":
            div.order = "attack"
            continue
        if doctrine == "strategic_reserve":
            div.order = "reserve"
            continue

        # WarAI tactical recommendation
        base_features = [
            div.soldiers / 1000.0,
            div.experience,
            div.equipment,
            nation.military / 100.0,
        ]
        state = base_features + ally_features + grid_features
        action = nation.military_ai.choose_action(state)
        raw_order = ["attack", "defend"][action]

        # Soft doctrine biases
        if doctrine == "defensive" and raw_order == "attack":
            div.order = "defend"   # defensive nations hold ground
        elif doctrine == "offensive" and raw_order == "defend":
            div.order = "attack"   # offensive nations push forward
        else:
            div.order = raw_order


def reward_divisions(divisions: List[Division], success: float) -> None:
    """Train division controllers based on battle ``success``.

    ``success`` should be positive for victory and negative for defeat.
    Successful divisions are trained toward their assigned order while
    unsuccessful ones are nudged toward a neutral reserve stance.
    """

    for div in divisions:
        context = [div.soldiers / 1000.0, div.experience, div.equipment]
        if success > 0:
            label = ORDER_TO_LABEL.get(div.order, ORDER_TO_LABEL["reserve"])
        else:
            label = ORDER_TO_LABEL["reserve"]
        div.controller.train([context], [label])


# ---------------------------------------------------------------------------
# City population helpers
# ---------------------------------------------------------------------------

def _deduct_city_population(nation: "Nation", amount: int) -> None:
    """Remove *amount* people from the nation's cities, spread proportionally.

    This makes recruitment and combat casualties persistent across turns
    because city populations are the source of ``nation.population``.
    """
    if amount <= 0 or not nation.cities:
        return
    total = sum(c.population for c in nation.cities)
    if total <= 0:
        return
    remaining = amount
    for city in nation.cities:
        if remaining <= 0:
            break
        share = int(amount * city.population / max(1, total))
        share = min(share, city.population, remaining)
        city.population = max(0, city.population - share)
        remaining -= share
    # Any rounding remainder comes from the largest city
    if remaining > 0:
        largest = max(nation.cities, key=lambda c: c.population)
        largest.population = max(0, largest.population - remaining)


def _apply_city_casualties(nation: "Nation", soldier_deaths: int) -> None:
    """Translate battlefield deaths into persistent city population loss."""
    _deduct_city_population(nation, soldier_deaths)


def build_division(nation: "Nation", template_name: Optional[str] = None) -> None:
    """Create a new division for ``nation`` using ``template_name``.

    Recruits are drawn from the city populations so the cost persists across
    turns.  ``nation.population`` is also decremented within the current turn
    so downstream budget checks in the same fifth stay consistent.
    """

    if not nation.division_templates:
        return
    if template_name is None:
        template_name = next(iter(nation.division_templates))
    tmpl = nation.division_templates.get(template_name)
    if not tmpl:
        return
    size = tmpl.base_size
    if nation.population < size:
        return
    # Deduct from city populations (persistent) and the within-turn total
    _deduct_city_population(nation, size)
    nation.population = max(0, nation.population - size)
    anchor = max(nation.cities, key=lambda c: c.population) if nation.cities else None
    x      = anchor.x      if anchor else 0
    y      = anchor.y      if anchor else 0
    planet = anchor.planet if anchor else nation.planet
    nation.divisions.append(
        Division(
            size,
            x,
            y,
            planet,
            tmpl.experience_mod,
            tmpl.equipment_mod,
            template_name,
            nation.doctrine,
        )
    )

def logistic_value(population: float, k: float, manpower_frac: float) -> float:
    """Return the logistic curve value — soldiers supportable at this population."""
    # S-curve from 0 to k*manpower_frac
    midpoint = k * 0.1  # inflection point at 10% of carrying capacity
    steepness = 0.00001  # tune this
    return (k * manpower_frac) / (1 + math.exp(-steepness * (population - midpoint)))


# ---------------------------------------------------------------------------
# Nuclear warfare
# ---------------------------------------------------------------------------

def produce_nuclear_weapons(nation: "Nation") -> None:
    """Convert uranium, metal and economy into nuclear stockpile."""
    if "Atomic Engineering" not in nation.tech_tree.unlocked:
        return
    uranium = nation.resources.get("uranium", 0.0)
    metal = nation.resources.get("metal", 0.0)
    base = int(min(uranium // 10, metal // 20, nation.economy_linear // 100))
    if base <= 0:
        return
    rate = sum(f.rate for f in nation.nuke_plants) if nation.nuke_plants else 1.0
    possible = int(
        min(
            base * rate,
            nation.resources.get("uranium", 0.0) // 10,
            nation.resources.get("metal", 0.0) // 20,
            nation.economy_linear // 100,
        )
    )
    if possible > 0:
        nation.nuclear_stockpile += possible
        nation.resources["uranium"] -= possible * 10
        nation.resources["metal"] -= possible * 20
        nation.economy_linear -= possible * 100


def launch_nuclear_strike(nation: "Nation", enemy: "Nation") -> None:
    """Inflict heavy losses on ``enemy`` using one warhead."""
    from ..planets import PLANETS

    nation.relations[enemy.id] = "enemy"
    enemy.relations[nation.id] = "enemy"
    nation.at_war.add(enemy.id)
    enemy.at_war.add(nation.id)

    nation.nuclear_stockpile -= 1

    tech_scale = 1.0 + nation.technology.military / 100.0
    defense = sum(d.strength for d in enemy.orbital_defenses)
    damp = max(0.0, 1.0 - defense)

    casualties = 0

    if enemy.cities:
        target = max(enemy.cities, key=lambda c: c.population)
        city_loss = int(target.population * 0.8 * tech_scale * damp)
        target.population = max(0, target.population - city_loss)
        infra_loss = max(1, int(target.infrastructure * 0.7 * damp))
        target.infrastructure = max(0, target.infrastructure - infra_loss)
        casualties += city_loss
    else:
        casualties += int(enemy.population * 0.2 * damp)

    enemy.population = max(0, enemy.population - casualties)
    enemy.economy_linear *= 0.3 * damp
    enemy.stability = max(0.0, enemy.stability - 40 * damp)
    nation.stability = max(0.0, nation.stability - 10)

    planet = PLANETS.get(enemy.planet)
    if planet:
        planet.radiation_level = min(1.0, planet.radiation_level + 0.4 * damp)

    wprint(nation.name, f"  {nation.name} launches a nuclear strike on {enemy.name}!")


def launch_first_strike(nation: "Nation", enemies: List["Nation"]) -> None:
    """Launch all available warheads across ``enemies`` in round-robin fashion."""
    if nation.nuclear_stockpile <= 0 or not enemies:
        return

    idx = 0
    while nation.nuclear_stockpile > 0:
        enemy = enemies[idx % len(enemies)]
        launch_nuclear_strike(nation, enemy)
        if enemy.nuclear_stockpile > 0 and random.random() < 0.5:
            launch_nuclear_strike(enemy, nation)
        idx += 1


def consider_first_strike(nation: "Nation", nations: Dict[int, "Nation"]) -> None:
    """Allow the military AI to initiate a nuclear first strike."""
    if not nation.military_ai or nation.nuclear_stockpile <= 0:
        return
    potential = [
        n
        for n in nations.values()
        if n.id != nation.id
        and n.id not in nation.at_war
        and nation.relations.get(n.id, "neutral") != "ally"
    ]
    if not potential:
        return
    state = [
        float(nation.nuclear_stockpile),
        nation.military,
        nation.economy,
        nation.stability,
        float(sum(e.nuclear_stockpile for e in potential)),
    ]
    act = nation.military_ai.choose_action(state)
    if act == 1:
        target = max(potential, key=lambda n: n.military)
        launch_first_strike(nation, [target])

def wage_war(
    nation: "Nation",
    nations: Dict[int, "Nation"],
    *,
    alliance_row: List[int] | None = None,
) -> None:
    """Run warfare calculations for ``nation`` against its enemies.

    Changes from earlier version
    ----------------------------
    * Division casualties are now cascaded to city populations via
      :func:`_apply_city_casualties`, making losses persist across turns.
    * The direct ``nation.population -= loss`` lines are removed; all
      demographic impact flows through city data.
    * War exhaustion is accumulated each fifth.  :func:`check_war_exhaustion`
      is consulted after combat and triggers :func:`make_peace` when either
      side reaches its breaking point.
    """
    from .diplomacy import (
        check_war_exhaustion, make_peace, accumulate_war_exhaustion,
        check_war_victory, enforce_victory, update_war_score,
    )

    enemies = [nations.get(eid) for eid in list(nation.at_war) if nations.get(eid)]
    if alliance_row is None and nations:
        matrix, _, index_map = build_alliance_matrix(nations)
        idx = index_map.get(nation.id)
        alliance_row = matrix[idx] if idx is not None else None
    issue_orders(nation, alliance_row)
    if (
        nation.nuclear_stockpile > 1
        and "Nuclear Weapons" in nation.tech_tree.unlocked
        and enemies
        and random.random() < 0.05
    ):
        launch_first_strike(nation, enemies)

    for enemy_id in list(nation.at_war):
        enemy = nations.get(enemy_id)
        if not enemy:
            nation.at_war.discard(enemy_id)
            continue
	
        loss_self = 0
        loss_enemy = 0
        engaged_self = 0.0
        engaged_enemy = 0.0
        stacked_self = 0.0
        stacked_enemy = 0.0
        sup_self: List[Division] = []
        sup_enemy: List[Division] = []
        for div in nation.divisions:
            move_division(div, nation, enemy, nations)
        issue_orders(enemy)
        p1 = PLANETS.get(nation.planet)
        p2 = PLANETS.get(enemy.planet)
        if not p1 or not p2:
            continue
        dist = distance(p1.coords, p2.coords)
        route_penalty = 0.0
        if p1 == p2 and nation.cities and enemy.cities:
            best = None
            for sc in nation.cities:
                for ec in enemy.cities:
                    d = p1.get_route_distance((sc.x, sc.y, "city"), (ec.x, ec.y, "city"))
                    if d != float("inf") and (best is None or d < best):
                        best = d
            route_penalty = 5.0 if best is None else best / 50.0
        cost = max(1.0, dist / 100 + route_penalty)
        bonus_self  = nation.tech_bonuses.get("division_power", 1.0)
        bonus_enemy = enemy.tech_bonuses.get("division_power", 1.0)

        enemy_planet  = PLANETS.get(enemy.planet)
        home_planet   = PLANETS.get(nation.planet)
        attack_supply = _supply_throughput(nation, enemy_planet)
        defend_supply = _supply_throughput(enemy, home_planet)
        readiness_self  = _combat_readiness(nation, attack_supply)
        def _supporting(n: "Nation", e: "Nation") -> List[Division]:
            if not n.divisions or not e.cities:
                return []
            sup: List[Division] = []
            for div in n.divisions:
                if div.planet != e.planet:
                    continue
                if div.decide_posture() == "reserve":
                    continue
                dist_near = min(((div.x - c.x) ** 2 + (div.y - c.y) ** 2) ** 0.5 for c in e.cities)
                if dist_near <= 100:
                    sup.append(div)
            return sup

        sup_self  = _supporting(nation, enemy)
        sup_enemy = _supporting(enemy, nation)
        stacked_self,  engaged_self  = _stacked_power(sup_self,  enemy_planet, bonus_self,  attack_supply)
        stacked_enemy, engaged_enemy = _stacked_power(sup_enemy, home_planet,  bonus_enemy, defend_supply)

        readiness_enemy = _combat_readiness(enemy,  defend_supply)
        power_self  = nation.military * readiness_self  + stacked_self
        power_enemy = enemy.military  * readiness_enemy + stacked_enemy
        ratio = max(power_self, power_enemy) / max(1.0, min(power_self, power_enemy))
        
        if (
            nation.nuclear_stockpile > 0
            and "Nuclear Weapons" in nation.tech_tree.unlocked
            and (power_enemy > power_self * 1.2 or nation.stability < 40)
            and random.random() < 0.3
        ):
            launch_nuclear_strike(nation, enemy)

        loss_self = int(
            cost * power_enemy / 200 * _attrition_multiplier(engaged_self,  attack_supply)
        )
        loss_enemy = int(
            cost * power_self  / 200 * _attrition_multiplier(engaged_enemy, defend_supply)
        )
        
        if ratio >= 10.0 and power_self > power_enemy:
            update_war_score(nation, enemy, kills_on_enemy=loss_enemy, kills_on_self=0)
            accumulate_war_exhaustion(enemy, nation.id, loss_enemy)
            if check_war_victory(nation, enemy):
                enforce_victory(nation, enemy, nations)
            continue
        significant = ratio >= 5 or abs(loss_self - loss_enemy) >= 1000
        if __debug__ and significant:
            wprint(
                nation.name,
                f"  {nation.name} vs {enemy.name}: "
                f"pwr={power_self:.0f}/{power_enemy:.0f}  "
                f"losses={loss_self}/{loss_enemy}",
            )

        # ------------------------------------------------------------------
        # Apply losses through division soldiers → city populations
        # ------------------------------------------------------------------
        nation.military = max(0.0, nation.military - cost)
        enemy.military  = max(0.0, enemy.military  - cost)

        success = 1.0 if power_self >= power_enemy else -1.0
        total_nation_kills = 0
        total_enemy_kills  = 0

        if sup_self:
            dispersal = 0.8 + 0.4 * min(1.5, attack_supply)
            kill_each = int(
                (loss_self * max(engaged_self, 0.0))
                / max(1, len(sup_self))
                / dispersal
            )
            for div in list(sup_self):
                kill = min(div.soldiers, kill_each)
                div.soldiers    -= kill
                total_nation_kills += kill
                if div.soldiers <= 0 and div in nation.divisions:
                    nation.divisions.remove(div)
            reward_divisions(sup_self, success)

        if sup_enemy:
            dispersal = 0.8 + 0.4 * min(1.5, defend_supply)
            kill_each = int(
                (loss_enemy * max(engaged_enemy, 0.0))
                / max(1, len(sup_enemy))
                / dispersal
            )
            for div in list(sup_enemy):
                kill = min(div.soldiers, kill_each)
                div.soldiers   -= kill
                total_enemy_kills += kill
                if div.soldiers <= 0 and div in enemy.divisions:
                    enemy.divisions.remove(div)
            reward_divisions(sup_enemy, -success)

        # Cascade division deaths into city populations (persistent loss)
        if total_nation_kills > 0:
            _apply_city_casualties(nation, total_nation_kills)
        if total_enemy_kills > 0:
            _apply_city_casualties(enemy, total_enemy_kills)

        # Small war overhead even when no divisions are engaged (supply drain,
        # civilian disruption, etc.) — capped at 0.5 % of population per fifth
        overhead_self  = min(int(nation.population * 0.005), max(0, loss_self  - total_nation_kills))
        overhead_enemy = min(int(enemy.population  * 0.005), max(0, loss_enemy - total_enemy_kills))
        if overhead_self  > 0:
            _apply_city_casualties(nation, overhead_self)
        if overhead_enemy > 0:
            _apply_city_casualties(enemy, overhead_enemy)

        # ------------------------------------------------------------------
        # Accumulate war exhaustion and update war score
        # ------------------------------------------------------------------
        accumulate_war_exhaustion(nation, enemy_id, total_nation_kills + overhead_self)
        accumulate_war_exhaustion(enemy,  nation.id, total_enemy_kills + overhead_enemy)

        # War score: positive for nation when it kills more than it loses
        update_war_score(
            nation, enemy,
            kills_on_enemy = total_enemy_kills  + overhead_enemy,
            kills_on_self  = total_nation_kills + overhead_self,
        )

        # ------------------------------------------------------------------
        # Resolve war outcome: annihilation > decisive victory > exhaustion
        # ------------------------------------------------------------------
        if enemy.population <= 0:
            enemy.killer_id = nation.id
            nation.at_war.discard(enemy_id)
            enemy.at_war.discard(nation.id)
            nation.war_exhaustion.pop(enemy_id, None)
            enemy.war_exhaustion.pop(nation.id, None)
            nation.war_score.pop(enemy_id, None)
            enemy.war_score.pop(nation.id, None)
            if __debug__:
                wprint(nation.name, f"  {nation.name} defeated {enemy.name}!")
        elif check_war_victory(nation, enemy):
            enforce_victory(nation, enemy, nations)
        elif check_war_victory(enemy, nation):
            enforce_victory(enemy, nation, nations)
        elif check_war_exhaustion(nation, enemy):
            make_peace(nation, enemy)

