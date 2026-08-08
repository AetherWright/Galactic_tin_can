"""War goals, war score tracking and victory enforcement."""

from __future__ import annotations

import random
from enum import Enum
from typing import Dict, TYPE_CHECKING

from ..core import wprint
from .peace import make_peace

if TYPE_CHECKING:
    from ..nations.nation import Nation


#: War score (bounded ±1) above which the leading side enforces its war goal.
WAR_SCORE_VICTORY: float = 0.65

#: Attacker power ratio ≥ this → TERRITORIAL goal.
_GOAL_TERRITORIAL_RATIO: float = 3.0
#: Attacker power ratio ≥ this → SUBJUGATION goal.
_GOAL_SUBJUGATION_RATIO: float = 2.0
#: Attacker power ratio ≥ this → HUMILIATION goal.
_GOAL_HUMILIATION_RATIO: float = 1.3

#: Probability that an ally joins a war its partner has entered.
ALLY_WAR_ENTRY_CHANCE: float = 0.40


# ---------------------------------------------------------------------------
# War goal enumeration
# ---------------------------------------------------------------------------

class WarGoal(str, Enum):
    """Possible victory conditions a belligerent nation pursues.

    BORDER_CLASH
        White peace — cease hostilities with no settlement.
    HUMILIATION
        Loser suffers a severe stability penalty; winner gains prestige.
    TERRITORIAL
        Winner annexes the loser's smallest / weakest city.
    SUBJUGATION
        Loser pays recurring tribute each fifth until the debt decays.
    """

    BORDER_CLASH = "border_clash"
    HUMILIATION  = "humiliation"
    TERRITORIAL  = "territorial"
    SUBJUGATION  = "subjugation"


# ---------------------------------------------------------------------------
# War goal assignment
# ---------------------------------------------------------------------------

def _nation_power(nation: "Nation") -> float:
    """Quick power estimate for goal assignment (no readiness multiplier)."""
    soldiers_norm = sum(d.soldiers for d in nation.divisions) / 1000.0
    return nation.military + soldiers_norm


def assign_war_goal(nation: "Nation", enemy: "Nation") -> str:
    """Choose a :class:`WarGoal` value for *nation* going to war with *enemy*.

    The goal is based on the rough power ratio at the moment of declaration:

    * ≥ 3 : 1 → :attr:`~WarGoal.TERRITORIAL`
    * ≥ 2 : 1 → :attr:`~WarGoal.SUBJUGATION`
    * ≥ 1.3 : 1 → :attr:`~WarGoal.HUMILIATION`
    * otherwise → :attr:`~WarGoal.BORDER_CLASH`
    """
    ratio = _nation_power(nation) / max(1.0, _nation_power(enemy))
    if ratio >= _GOAL_TERRITORIAL_RATIO:
        return WarGoal.TERRITORIAL.value
    if ratio >= _GOAL_SUBJUGATION_RATIO:
        return WarGoal.SUBJUGATION.value
    if ratio >= _GOAL_HUMILIATION_RATIO:
        return WarGoal.HUMILIATION.value
    return WarGoal.BORDER_CLASH.value


# ---------------------------------------------------------------------------
# War score tracking
# ---------------------------------------------------------------------------

def update_war_score(
    nation:         "Nation",
    enemy:          "Nation",
    kills_on_enemy: int,
    kills_on_self:  int,
) -> None:
    """Adjust both nations' war score based on the latest exchange.

    Positive score = nation is winning; negative = nation is losing.
    Score is bounded to [-1, 1].  Both sides are updated symmetrically.
    """
    pop_scale = max(1000, nation.population // 100)
    net   = kills_on_enemy - kills_on_self
    power_ratio = nation.military / max(1.0, enemy.military)
    dominance_bonus = min(0.02, (power_ratio - 1.0) * 0.002) if power_ratio > 2.0 else 0.0
    delta = max(-0.08, min(0.08, net / pop_scale)) + dominance_bonus
    nation.war_score[enemy.id] = max(
        -1.0, min(1.0, nation.war_score.get(enemy.id, 0.0) + delta)
    )
    # Enemy's score mirrors ours
    enemy.war_score[nation.id] = max(
        -1.0, min(1.0, enemy.war_score.get(nation.id, 0.0) - delta)
    )


def check_war_victory(nation: "Nation", enemy: "Nation") -> bool:
    """Return ``True`` when *nation* has achieved a decisive advantage."""
    return nation.war_score.get(enemy.id, 0.0) >= WAR_SCORE_VICTORY


# ---------------------------------------------------------------------------
# Victory enforcement
# ---------------------------------------------------------------------------

def enforce_victory(
    victor: "Nation",
    loser:  "Nation",
    nations: Dict[int, "Nation"],
) -> None:
    """Apply the victor's war goal against the loser, then end the war.

    Goal effects
    ------------
    BORDER_CLASH
        Simple white peace — no additional effect.
    HUMILIATION
        Loser suffers −25 stability; victor gains +10 stability and +5 military.
    TERRITORIAL
        Victor annexes the loser's least-populated city.
    SUBJUGATION
        Loser is placed in tribute to the victor (8 % of current economy/fifth,
        decaying 1 % per fifth until it falls below 1.0).
    """
    goal = victor.war_goals.get(loser.id, WarGoal.BORDER_CLASH.value)

    wprint(
        victor.name,
        f"  {victor.name} enforces \u2018{goal}\u2019 on {loser.name}!",
    )

    if goal == WarGoal.HUMILIATION.value:
        loser.stability   = max(0.0, loser.stability  - 25.0)
        victor.stability  = min(100.0, victor.stability + 10.0)
        victor.military  += 5.0

    elif goal == WarGoal.TERRITORIAL.value:
        if loser.cities:
            taken = min(loser.cities, key=lambda c: c.population)
            taken.owner = victor.id
            loser.cities.remove(taken)
            victor.cities.append(taken)
            victor.infrastructure += 5
            wprint(
                victor.name,
                f"    {victor.name} annexes city at {taken.coords} from {loser.name}",
            )

    elif goal == WarGoal.SUBJUGATION.value:
        tribute = loser.economy_linear * 0.08
        if tribute > 0:
            loser.tribute_debts[victor.id] = tribute

    # All goals end with peace (label reflects the decisive outcome)
    make_peace(victor, loser, label=f"{goal} victory")


# ---------------------------------------------------------------------------
# Ally war entry
# ---------------------------------------------------------------------------

#: Floor on the war-weariness multiplier (see _war_weariness_factor) — entry
#: is dampened as more of the galaxy is already fighting, but never fully
#: closed off even at maximum galaxy-wide conflict.
_MIN_WAR_WEARINESS_FACTOR: float = 0.15


def _war_weariness_factor(nations: Dict[int, "Nation"]) -> float:
    """Return a [0.15..1.0] multiplier on ally war-entry chance that shrinks
    as more of the galaxy is already at war.

    Without this, every qualifying ally rolls the same flat
    :data:`ALLY_WAR_ENTRY_CHANCE` regardless of how much of the galaxy is
    already burning — so one war can chain through alliance webs into a
    galaxy-wide conflagration at a constant rate. Scaling the chance down as
    galaxy-wide conflict intensity rises makes remaining neutral nations
    progressively more reluctant to pile on, which self-dampens cascades
    instead of amplifying them.
    """
    if not nations:
        return 1.0
    at_war_fraction = sum(1 for n in nations.values() if n.at_war) / len(nations)
    return max(_MIN_WAR_WEARINESS_FACTOR, 1.0 - at_war_fraction)


def notify_ally_war_entry(
    nation:  "Nation",
    enemy:   "Nation",
    nations: Dict[int, "Nation"],
) -> None:
    """Give *nation*'s allies the chance to join the war against *enemy*.

    An ally will enter the war if:
    * It is not already at war with anyone (avoid spreading too thin).
    * Its doctrine is not ``strategic_reserve`` or ``economic``.
    * A random draw is below :data:`ALLY_WAR_ENTRY_CHANCE`, scaled down by
      :func:`_war_weariness_factor` as more of the galaxy is already fighting.
    """
    weariness = _war_weariness_factor(nations)
    for ally_id in list(nation.alliances):
        if ally_id == enemy.id:
            continue
        ally = nations.get(ally_id)
        if not ally:
            continue
        if enemy.id in ally.at_war or ally_id in enemy.at_war:
            continue   # already fighting
        if ally.at_war:
            continue   # already in a separate war — don't drag in
        if ally.doctrine_signal in ("strategic_reserve", "economic"):
            continue   # pacifist doctrine; stay out
        if random.random() >= ALLY_WAR_ENTRY_CHANCE * weariness:
            continue
        if ally.planet != enemy.planet:
            continue
        # Ally enters the war
        ally.at_war.add(enemy.id)
        enemy.at_war.add(ally_id)
        ally.war_goals[enemy.id]    = WarGoal.BORDER_CLASH.value
        ally.war_exhaustion.pop(enemy.id, None)
        enemy.war_exhaustion.pop(ally_id, None)
        wprint(
            nation.name,
            f"  {ally.name} joins the war against {enemy.name} "
            f"(allied with {nation.name})",
        )
