"""Trade bonuses between partners and tribute collection."""

from __future__ import annotations

from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ..nations.nation import Nation


#: Economy fraction gained per fifth from an *allied* trade partner.
ALLY_TRADE_RATE: float = 0.08

#: Economy fraction gained per fifth from a non-allied trade partner.
PARTNER_TRADE_RATE: float = 0.05


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
