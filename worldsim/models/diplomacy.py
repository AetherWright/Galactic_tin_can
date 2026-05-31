"""Diplomatic relations, communication, and peace negotiation for WorldSim.

This module owns all inter-nation soft-power and messaging logic so that
``nation.py`` stays focused on economic and demographic simulation.

Public API — messaging
----------------------
Message
    Communication packet placed in a nation's inbox.
can_communicate(nation, other) → bool
send_message / process_messages / supply_nation

Public API — diplomacy & trade
-------------------------------
update_border_pressure(nation, nations, bloc_map) → None
compute_trade_bonus(nation, nations) → float
    Higher rate for allied trade partners (ALLY_TRADE_RATE)
    vs plain trade partners (PARTNER_TRADE_RATE).
collect_tributes(nation, nations) → None
    Pay/receive per-fifth tribute debts set by war settlements.

Public API — war goals & victory
---------------------------------
WarGoal
    Enum of settlement types (BORDER_CLASH, HUMILIATION, TERRITORIAL,
    SUBJUGATION).
assign_war_goal(nation, enemy) → str
    Pick a goal based on attacker power ratio at war declaration.
update_war_score(nation, enemy, kills_on_enemy, kills_on_self) → None
    Adjust war score (bounded ±1) symmetrically for both sides.
check_war_victory(nation, enemy) → bool
    True when nation's war score >= WAR_SCORE_VICTORY (0.65).
enforce_victory(victor, loser, nations) → None
    Apply the victor's war goal then call make_peace.
notify_ally_war_entry(nation, enemy, nations) → None
    Give allies a chance to join the war.

Public API — exhaustion / ceasefire
------------------------------------
check_war_exhaustion / make_peace
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from ..utils import distance, wprint

if TYPE_CHECKING:
    from .nation import Nation
    from .war import AllianceBloc


# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

#: War exhaustion score at which a nation starts looking for peace exits.
PEACE_EXHAUSTION_THRESHOLD: float = 0.60

#: Stability floor below which an exhausted nation will always seek peace.
PEACE_STABILITY_FLOOR: float = 35.0

#: Stability cost to both parties when a war ends in a negotiated peace.
PEACE_STABILITY_COST: float = 2.0

#: Maximum star-distance that allows direct communication.
COMM_RANGE_LY: float = 15.0

#: Power ratio above which the dominant side can impose a peace settlement.
DOMINANCE_RATIO: float = 8.0

# --- War goals & scoring ---

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

# --- Trade rates ---

#: Economy fraction gained per fifth from an *allied* trade partner.
ALLY_TRADE_RATE: float = 0.08

#: Economy fraction gained per fifth from a non-allied trade partner.
PARTNER_TRADE_RATE: float = 0.05


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
# Message
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """Communication packet exchanged between nations.

    Parameters
    ----------
    sender:
        Nation id of the sender.
    text:
        Human-readable description.
    kind:
        Semantic label used by AI to trigger behaviour.
    payload:
        Optional data for ``kind``-specific actions.
    """

    sender: int
    text: str
    kind: str = "text"
    payload: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Alliance helpers
# ---------------------------------------------------------------------------

def get_all_allies(nation: "Nation", nations: Dict[int, "Nation"]) -> Set[int]:
    """Return the transitive closure of all allied nation ids (including self)."""
    visited: Set[int] = set()
    stack = [nation.id]
    while stack:
        cur = stack.pop()
        if cur in visited or cur not in nations:
            continue
        visited.add(cur)
        for ally in nations[cur].alliances:
            if ally not in visited:
                stack.append(ally)
    return visited


# ---------------------------------------------------------------------------
# Communication helpers
# ---------------------------------------------------------------------------

def can_communicate(nation: "Nation", other: "Nation") -> bool:
    """Return ``True`` if the nations are within communication range."""
    from .nation import STARS

    my_stars    = [s for s in STARS.values() if s.owner == nation.id]
    other_stars = [s for s in STARS.values() if s.owner == other.id]
    if my_stars and other_stars:
        edge_dist = min(
            s1.distance_to(s2) for s1 in my_stars for s2 in other_stars
        )
        return edge_dist <= COMM_RANGE_LY
    return distance(nation.world_centroid, other.world_centroid) <= COMM_RANGE_LY


def send_message(
    nation: "Nation",
    other: "Nation",
    text: str,
    *,
    kind: str = "text",
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    """Append a :class:`Message` to *other*'s inbox if in range."""
    if can_communicate(nation, other):
        other.inbox.append(Message(nation.id, text, kind, payload))


def process_messages(
    nation: "Nation",
    nations: Dict[int, "Nation"],
) -> None:
    """React to queued messages, then clear the inbox."""
    for msg in list(nation.inbox):
        nation.stability = min(100.0, nation.stability + 0.5)
        if msg.kind == "supply" and msg.payload:
            nation.economy_linear += msg.payload.get("economy", 0.0)
            nation.military      += msg.payload.get("military", 0.0)
        elif msg.kind == "support_request":
            sender = nations.get(msg.sender)
            if sender:
                econ = nation.economy_linear * 0.1
                mil  = nation.military * 0.1
                nation.economy_linear -= econ
                nation.military       -= mil
                send_message(
                    nation,
                    sender,
                    "Supply shipment",
                    kind    = "supply",
                    payload = {"economy": econ, "military": mil},
                )
        elif msg.kind == "peace_offer":
            # Accept automatically when exhausted or severely weakened
            sender = nations.get(msg.sender)
            if sender and _should_accept_peace(nation, sender.id):
                make_peace(nation, sender)
    nation.inbox.clear()


def supply_nation(
    nation: "Nation",
    other: "Nation",
    *,
    military: float = 0.0,
    economy: float = 0.0,
) -> None:
    """Transfer resources to *other* without entering a war."""
    nation.military      = max(0.0, nation.military - military)
    nation.economy_linear = max(0.0, nation.economy_linear - economy)
    send_message(
        nation,
        other,
        "Supply shipment",
        kind    = "supply",
        payload = {"military": military, "economy": economy},
    )


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
            reward = nation.compute_reward("diplomacy", state, new_state)
            nation.diplomacy_ai.train(state, act, reward, new_state)

    # Border pressure bleeds stability
    for oid, val in list(nation.border_pressure.items()):
        if val > 1.0:
            nation.stability -= val * 0.05


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

def notify_ally_war_entry(
    nation:  "Nation",
    enemy:   "Nation",
    nations: Dict[int, "Nation"],
) -> None:
    """Give *nation*'s allies the chance to join the war against *enemy*.

    An ally will enter the war if:
    * It is not already at war with anyone (avoid spreading too thin).
    * Its doctrine is not ``strategic_reserve`` or ``economic``.
    * A random draw is below :data:`ALLY_WAR_ENTRY_CHANCE` (40 %).
    """
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
        if random.random() >= ALLY_WAR_ENTRY_CHANCE:
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


# ---------------------------------------------------------------------------
# Trade helpers
# ---------------------------------------------------------------------------

def compute_trade_bonus(
    nation:  "Nation",
    nations: Dict[int, "Nation"],
) -> float:
    """Return the total per-fifth trade bonus from all trade partners.

    Partners who are also alliance members receive :data:`ALLY_TRADE_RATE`
    (8 %); plain trade partners receive :data:`PARTNER_TRADE_RATE` (5 %).
    """
    bonus = 0.0
    for pid in nation.trade_partners:
        partner = nations.get(pid)
        if not partner:
            continue
        rate = ALLY_TRADE_RATE if pid in nation.alliances else PARTNER_TRADE_RATE
        bonus += partner.economy_linear * rate
    return min(bonus, nation.economy_linear * 2.0)


def collect_tributes(
    nation:  "Nation",
    nations: Dict[int, "Nation"],
) -> None:
    """Process tribute payments owed by *nation* to its creditors.

    Each debt is capped at 5 % of current economy per fifth and decays
    2 % per fifth.  Payments are suspended while the debtor is at war
    with the creditor (to avoid paying the enemy during active conflict).
    """
    for creditor_id, amount in list(nation.tribute_debts.items()):
        creditor = nations.get(creditor_id)
        if not creditor:
            del nation.tribute_debts[creditor_id]
            continue
        # Suspend payment during active hostilities
        if creditor_id in nation.at_war:
            continue
        actual = min(amount, nation.economy_linear * 0.05)
        if actual > 0:
            nation.economy_linear    = max(0.0, nation.economy_linear - actual)
            creditor.economy_linear += actual
        # Decay: 2 % per fifth
        new_amount = amount * 0.98
        if new_amount < 1.0:
            del nation.tribute_debts[creditor_id]
        else:
            nation.tribute_debts[creditor_id] = new_amount
