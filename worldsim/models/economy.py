from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
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
    """

    funds: float = 100.0
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
