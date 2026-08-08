"""Border pressure dynamics and the diplomacy AI decision loop."""

from __future__ import annotations

from typing import Dict, TYPE_CHECKING

from ..core import distance
from .wargoals import assign_war_goal, notify_ally_war_entry

if TYPE_CHECKING:
    from ..military.alliances import AllianceBloc
    from ..nations.nation import Nation


# ---------------------------------------------------------------------------
# Border pressure & diplomacy AI
# ---------------------------------------------------------------------------

def update_border_pressure(
    nation: "Nation",
    nations: Dict[int, "Nation"],
    bloc_map: Dict[int, "AllianceBloc"],
) -> None:
    """Adjust border pressure for all neighbours and run the diplomacy AI."""
    self_bloc   = bloc_map.get(nation.id)
    my_centroid = self_bloc.centroid if self_bloc else nation.world_centroid
    my_military = self_bloc.military if self_bloc else nation.military

    for other in nations.values():
        if other.id == nation.id:
            continue
        other_bloc = bloc_map.get(other.id)
        if self_bloc and other_bloc and self_bloc.id == other_bloc.id:
            nation.border_pressure[other.id] = 0.0
            continue

        cur = nation.border_pressure.get(other.id, 0.0)
        if not nation.cities or not other.cities:
            nation.border_pressure[other.id] = max(0.0, cur - 0.1)
            continue
        if nation.planet != other.planet:
            nation.border_pressure[other.id] = max(0.0, cur - 0.05)
            continue

        other_centroid = other_bloc.centroid if other_bloc else other.world_centroid
        dist = distance(my_centroid, other_centroid)
        if dist > 150:
            nation.border_pressure[other.id] = max(0.0, cur - 0.05)
            continue

        other_mil = other_bloc.military if other_bloc else other.military
        pressure  = (other_mil - my_military) / 200.0
        pressure += max(0.0, 150 - dist) / 150
        pop_ratio  = other.population / max(1.0, nation.population)
        pressure += (1 - pop_ratio) * 0.5
        val = max(0.0, pressure)
        nation.border_pressure[other.id] = val

        # Diplomacy AI: choose ally / declare war / do nothing
        if nation.diplomacy_ai and other.id != nation.id:
            state = [
                val,
                nation.military / max(1.0, other.military),
                nation.economy  / max(1.0, other.economy),
                nation.stability / 100.0,
                1.0 if other.id in nation.alliances else 0.0,
            ]
            act = nation.diplomacy_ai.choose_action(state)
            reward_before = nation.reward_snapshot()

            if act == 1 and other.id not in nation.alliances and other.id not in nation.at_war:
                # Form alliance — also establish trade partnership
                nation.alliances.add(other.id)
                other.alliances.add(nation.id)
                nation.trade_partners.add(other.id)
                other.trade_partners.add(nation.id)
                nation.relations[other.id] = "ally"
                other.relations[nation.id] = "ally"
            elif act == 2 and other.id not in nation.at_war:
                # Declare war — only when not already allies
                if other.id not in nation.alliances:
                    nation.at_war.add(other.id)
                    other.at_war.add(nation.id)
                    nation.war_exhaustion.pop(other.id, None)
                    other.war_exhaustion.pop(nation.id, None)
                    # Assign war goals and notify allies on both sides
                    nation.war_goals[other.id] = assign_war_goal(nation, other)
                    notify_ally_war_entry(nation, other, nations)
                    notify_ally_war_entry(other, nation, nations)

            new_state = [
                nation.border_pressure.get(other.id, 0.0),
                nation.military / max(1.0, other.military),
                nation.economy  / max(1.0, other.economy),
                nation.stability / 100.0,
                1.0 if other.id in nation.alliances else 0.0,
            ]
            reward = nation.compute_reward(reward_before, nation.reward_snapshot())
            nation.diplomacy_ai.train(state, act, reward, new_state)

    # Border pressure bleeds stability
    for oid, val in list(nation.border_pressure.items()):
        if val > 1.0:
            nation.stability -= val * 0.05
