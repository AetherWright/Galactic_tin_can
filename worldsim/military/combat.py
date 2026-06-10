"""Ground-combat resolution: orders and the per-fifth war loop."""

from __future__ import annotations

import random
from typing import Dict, List, TYPE_CHECKING

from ..ai.representations import build_alliance_matrix, galactic_control_layers
from ..core import distance, wprint
from ..planets import PLANETS
from .divisions import Division, move_division, reward_divisions
from .logistics import (
    _attrition_multiplier,
    _combat_readiness,
    _stacked_power,
    _supply_throughput,
)
from .nuclear import launch_first_strike, launch_nuclear_strike

if TYPE_CHECKING:
    from ..nations.nation import Nation


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
    from ..diplomacy import (
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

