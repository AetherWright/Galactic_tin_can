"""War exhaustion accumulation and peace negotiation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core import wprint

if TYPE_CHECKING:
    from ..nations.nation import Nation


#: War exhaustion score at which a nation starts looking for peace exits.
PEACE_EXHAUSTION_THRESHOLD: float = 0.60

#: Stability floor below which an exhausted nation will always seek peace.
PEACE_STABILITY_FLOOR: float = 35.0

#: Stability cost to both parties when a war ends in a negotiated peace.
PEACE_STABILITY_COST: float = 2.0

#: Power ratio above which the dominant side can impose a peace settlement.
DOMINANCE_RATIO: float = 8.0


# ---------------------------------------------------------------------------
# War ending — exhaustion & peace
# ---------------------------------------------------------------------------

def accumulate_war_exhaustion(
    nation: "Nation",
    enemy_id: int,
    soldier_losses: int,
) -> None:
    """Increase *nation*'s exhaustion for its war against *enemy_id*.

    Exhaustion grows with actual manpower losses and ticks up a small flat
    amount each fifth even without casualties, modelling political fatigue.
    """
    pop = max(1, nation.population)
    gain = (soldier_losses / pop) * 50.0 + 0.004   # 0.4 % + time increment
    nation.war_exhaustion[enemy_id] = (
        nation.war_exhaustion.get(enemy_id, 0.0) + gain
    )


def _should_accept_peace(nation: "Nation", enemy_id: int) -> bool:
    """Return True when *nation* would rationally accept peace with *enemy_id*."""
    exhaustion = nation.war_exhaustion.get(enemy_id, 0.0)
    if exhaustion >= PEACE_EXHAUSTION_THRESHOLD and nation.stability <= PEACE_STABILITY_FLOOR:
        return True
    total_soldiers = sum(d.soldiers for d in nation.divisions)
    if total_soldiers == 0 and nation.military < 2.0:
        return True
    return False


def check_war_exhaustion(nation: "Nation", enemy: "Nation") -> bool:
    """Return True if *either* side should seek peace right now.

    Triggers when:
    * One side's exhaustion exceeds the threshold **and** stability is low.
    * One side has no remaining military force at all.
    * The power imbalance is overwhelming (8:1) and the weaker side is tired.
    """
    if _should_accept_peace(nation, enemy.id) or _should_accept_peace(enemy, nation.id):
        return True

    # Dominant-power imposed peace: one side can simply crush further fighting
    nation_power = nation.military + sum(d.soldiers / 1000 for d in nation.divisions)
    enemy_power  = enemy.military  + sum(d.soldiers / 1000 for d in enemy.divisions)
    if nation_power > 0 and enemy_power > 0:
        ratio = max(nation_power, enemy_power) / min(nation_power, enemy_power)
        weaker_exhaustion = (
            enemy.war_exhaustion.get(nation.id, 0.0)
            if nation_power > enemy_power
            else nation.war_exhaustion.get(enemy.id, 0.0)
        )
        if ratio >= DOMINANCE_RATIO and weaker_exhaustion > 0.20:
            return True
    return False


def make_peace(
    nation: "Nation",
    enemy:  "Nation",
    *,
    label: str = "ceasefire",
) -> None:
    """End the war between *nation* and *enemy*.

    Both parties pay a small stability cost, clear their exhaustion,
    war score, and war goals for each other, then relations go neutral.
    """
    nation.at_war.discard(enemy.id)
    enemy.at_war.discard(nation.id)

    # Clear war accounting
    nation.war_exhaustion.pop(enemy.id, None)
    enemy.war_exhaustion.pop(nation.id, None)
    nation.war_score.pop(enemy.id, None)
    enemy.war_score.pop(nation.id, None)
    nation.war_goals.pop(enemy.id, None)
    enemy.war_goals.pop(nation.id, None)

    # Small stability hit — war is costly even when it ends
    nation.stability = max(0.0, nation.stability - PEACE_STABILITY_COST)
    enemy.stability  = max(0.0, enemy.stability  - PEACE_STABILITY_COST)

    # Normalise relations
    nation.relations[enemy.id] = "neutral"
    enemy.relations[nation.id] = "neutral"

    wprint(
        nation.name,
        f"  Peace: {nation.name} \u2194 {enemy.name} ({label})",
    )
