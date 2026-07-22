from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Dict
import math


@dataclass(slots=True)
class Economy:
    """Tracks national funds and resource stockpiles.

    Parameters
    ----------
    funds:
        Generic economic value used for construction and upkeep.
    resources:
        Material stockpiles such as food or metal.
    caps:
        Maximum storage capacity for each resource. Growth slows as a
        stockpile approaches its cap to avoid abrupt overflow.
    reserves:
        Accumulated treasury (linear).  Each turn's net income is banked here
        and it is drawn down by civilian construction via the soft action
        budget.  Capped relative to gross output so it can't grow unbounded.
    last_net_income:
        Net income (gross output minus upkeep) computed on the most recent
        settled turn.  Feeds the per-turn civilian build budget.
    """

    # How much of the standing reserve a nation is willing to commit to civilian
    # construction in a single turn, on top of the turn's net income.
    _RESERVE_DRAW:     ClassVar[float] = 0.34
    # Treasury ceiling, as a multiple of the turn's gross output.
    _RESERVE_CAP_MULT: ClassVar[float] = 5.0

    funds: float = 100.0
    reserves: float = 0.0
    last_net_income: float = 0.0
    resources: Dict[str, float] = field(
        default_factory=lambda: {
            "food": 100.0,
            "metal": 100.0,
            "energy": 100.0,
            "uranium": 0.0,
        }
    )
    caps: Dict[str, float] = field(
        default_factory=lambda: {
            "food": 1000.0,
            "metal": 1000.0,
            "energy": 1000.0,
            "uranium": 500.0,
        }
    )

    def __post_init__(self) -> None:
        # Store the economy internally on a logarithmic scale so that
        # exceptionally wealthy nations experience diminishing marginal
        # returns.  The public API exposes helpers for working with the
        # underlying linear value.
        self.funds = self._to_log(self.funds)

    # Internal helpers -------------------------------------------------
    @staticmethod
    def _to_log(value: float) -> float:
        return math.log1p(max(0.0, value))

    @staticmethod
    def _from_log(value: float) -> float:
        return max(0.0, math.expm1(min(value, 709.0)))

    # Public helpers ---------------------------------------------------
    @property
    def linear_funds(self) -> float:
        """Return the underlying linear currency value."""

        return self._from_log(self.funds)

    def set_linear_funds(self, value: float) -> None:
        """Set ``funds`` using a linear value instead of the log scale."""

        self.funds = self._to_log(value)

    def add(self, amount: float) -> None:
        self.set_linear_funds(self.linear_funds + amount)

    def spend(self, amount: float) -> float:
        actual = self.linear_funds
        spent = min(actual, amount)
        self.set_linear_funds(actual - spent)
        return spent

    # Treasury / budget ------------------------------------------------
    def settle_turn(self, gross_output: float, upkeep: float) -> float:
        """Bank this turn's net income into the reserve and record it.

        ``net = gross_output - upkeep`` (upkeep floored at 0).  The net is
        added to :attr:`reserves` (which is floored at 0 and capped relative
        to gross output so the treasury can't grow without bound).  Returns
        the net income for the turn.
        """
        net = float(gross_output) - max(0.0, float(upkeep))
        self.last_net_income = net
        self.reserves = max(0.0, self.reserves + net)
        cap = self._RESERVE_CAP_MULT * max(float(gross_output), 1.0)
        if self.reserves > cap:
            self.reserves = cap
        return net

    def civilian_budget(self) -> float:
        """Spendable civilian-construction budget for the current turn.

        The turn's net income plus a fixed fraction of the standing reserve —
        a nation spends roughly what it earns, dipping into savings for a
        bigger building push when it has them.
        """
        return max(0.0, self.last_net_income) + self._RESERVE_DRAW * max(0.0, self.reserves)

    def draw_reserve(self, amount: float) -> None:
        """Deduct *amount* of committed construction spend from the reserve."""
        self.reserves = max(0.0, self.reserves - max(0.0, float(amount)))

    def has_resources(self, cost: Dict[str, float]) -> bool:
        return all(self.resources.get(k, 0.0) >= v for k, v in cost.items())

    def spend_resources(self, cost: Dict[str, float]) -> None:
        for k, v in cost.items():
            self.resources[k] = max(0.0, self.resources.get(k, 0.0) - v)

    def add_resource(self, kind: str, amount: float) -> None:
        cap = self.caps.get(kind)
        current = self.resources.get(kind, 0.0)
        if not cap or cap <= 0:
            self.resources[kind] = current + amount
            return
        remaining_ratio = max(0.0, 1 - current / cap)
        adjusted = amount * remaining_ratio
        self.resources[kind] = min(cap, current + adjusted)
