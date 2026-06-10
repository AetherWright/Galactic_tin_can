"""The diplomacy arm: soft power between nations.

Submodules
----------
``messages``   communication packets, range checks, supply shipments
``relations``  border pressure dynamics and the diplomacy AI loop
``wargoals``   war goals, war score and victory enforcement
``peace``      war exhaustion and peace negotiation
``trade``      trade-partner bonuses and tribute collection
"""

from .messages import (
    COMM_RANGE_LY,
    Message,
    can_communicate,
    get_all_allies,
    process_messages,
    send_message,
    supply_nation,
)
from .peace import (
    DOMINANCE_RATIO,
    PEACE_EXHAUSTION_THRESHOLD,
    PEACE_STABILITY_COST,
    PEACE_STABILITY_FLOOR,
    accumulate_war_exhaustion,
    check_war_exhaustion,
    make_peace,
)
from .relations import update_border_pressure
from .trade import (
    ALLY_TRADE_RATE,
    PARTNER_TRADE_RATE,
    collect_tributes,
    compute_trade_bonus,
)
from .wargoals import (
    ALLY_WAR_ENTRY_CHANCE,
    WAR_SCORE_VICTORY,
    WarGoal,
    assign_war_goal,
    check_war_victory,
    enforce_victory,
    notify_ally_war_entry,
    update_war_score,
)

__all__ = [
    "Message",
    "COMM_RANGE_LY",
    "can_communicate",
    "get_all_allies",
    "send_message",
    "process_messages",
    "supply_nation",
    "update_border_pressure",
    "accumulate_war_exhaustion",
    "check_war_exhaustion",
    "make_peace",
    "PEACE_EXHAUSTION_THRESHOLD",
    "PEACE_STABILITY_FLOOR",
    "PEACE_STABILITY_COST",
    "DOMINANCE_RATIO",
    "WarGoal",
    "assign_war_goal",
    "update_war_score",
    "check_war_victory",
    "enforce_victory",
    "notify_ally_war_entry",
    "WAR_SCORE_VICTORY",
    "ALLY_WAR_ENTRY_CHANCE",
    "compute_trade_bonus",
    "collect_tributes",
    "ALLY_TRADE_RATE",
    "PARTNER_TRADE_RATE",
]
