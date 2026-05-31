"""Government archetype model, national projects, and infrastructure cost tables."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, TYPE_CHECKING

from ..culture import ARCHETYPE_BONUSES, ArchetypeBonus
from ..config import load_json

if TYPE_CHECKING:
    from .nation import Nation
    from ..culture import Culture


@dataclass(slots=True)
class Government:
    """Archetype mixing weights derived from culture proximity."""

    weights: Dict[str, float] = field(
        default_factory=lambda: {k: 1.0 / 8 for k in ARCHETYPE_BONUSES}
    )
    approval: float = 60.0

    def update_weights(self, culture: "Culture") -> None:
        """Recompute archetype weights from culture trait proximity."""
        from ..culture import ARCHETYPE_IDEALS

        distances = {}
        culture_vec = list(culture.asdict().values())
        for name, ideal in ARCHETYPE_IDEALS.items():
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(culture_vec, ideal)))
            distances[name] = dist

        inv = {k: 1.0 / (d + 1e-6) for k, d in distances.items()}
        total = sum(inv.values())
        self.weights = {k: v / total for k, v in inv.items()}

    def effective_bonus(self) -> ArchetypeBonus:
        """Return a weighted blend of all archetype bonuses."""
        result = ArchetypeBonus()
        for name, weight in self.weights.items():
            bonus = ARCHETYPE_BONUSES[name]
            result.stability_flat += bonus.stability_flat * weight
            result.military_flat += bonus.military_flat * weight
            result.diplomacy_mult *= bonus.diplomacy_mult ** weight
            result.trade_mult *= bonus.trade_mult ** weight
            result.economy_mult *= bonus.economy_mult ** weight
            result.science_mult *= bonus.science_mult ** weight
            result.plague_resist += bonus.plague_resist * weight
            result.ship_cost_mult *= bonus.ship_cost_mult ** weight
            result.tribute_mult *= bonus.tribute_mult ** weight
            result.stability_decay_at_peace += bonus.stability_decay_at_peace * weight
            result.stability_decay_no_expansion += bonus.stability_decay_no_expansion * weight
            result.stability_scale_per_star += bonus.stability_scale_per_star * weight
            result.ally_trade_bonus += bonus.ally_trade_bonus * weight
            result.civil_war_rebel_strength += (bonus.civil_war_rebel_strength - 1.0) * weight
        result.civil_war_rebel_strength = max(1.0, result.civil_war_rebel_strength)
        return result

    def dominant_archetype(self) -> str:
        """Return the archetype with highest weight."""
        return max(self.weights, key=lambda k: self.weights[k])

    def bonuses(self) -> Dict[str, float]:
        """Backwards compatible with existing bonuses() calls."""
        bonus = self.effective_bonus()
        return {
            "economy": bonus.economy_mult,
            "stability": bonus.stability_flat,
            "military": bonus.military_flat,
        }


@dataclass(slots=True)
class NationalProject:
    """Large-scale construction tracked at the nation level."""

    name: str
    cost: float
    progress: float = 0.0
    on_complete: Optional[callable] = None
    prereqs: Set[str] = field(default_factory=set)

    def advance(self, amount: float) -> bool:
        """Increase progress by ``amount`` and return ``True`` if finished."""
        remaining = max(self.cost - self.progress, 0.0)
        factor = remaining / self.cost if self.cost else 1.0
        self.progress += amount * factor
        return self.progress >= self.cost


@dataclass(slots=True)
class ProjectSpec:
    """Specification for a buildable national project."""

    cost: float
    on_complete: callable
    prereqs: Set[str] = field(default_factory=set)


PROJECT_CATALOG: Dict[str, ProjectSpec] = {
    "Highway Network": ProjectSpec(
        100.0,
        lambda n: setattr(n, "infrastructure", n.infrastructure + 20),
    ),
    "Research Complex": ProjectSpec(
        80.0,
        lambda n: setattr(
            n.technology, "science", min(100.0, n.technology.science + 10.0)
        ),
        {"Highway Network"},
    ),
    "Orbital Defense Grid": ProjectSpec(
        120.0,
        lambda n: setattr(n, "military", n.military + 20),
    ),
    "Mega Dam": ProjectSpec(
        90.0,
        lambda n: (
            setattr(n, "infrastructure", n.infrastructure + 15),
            setattr(n, "economy_linear", n.economy_linear + 20),
        ),
        {"Highway Network"},
    ),
    "AI Governance System": ProjectSpec(
        110.0,
        lambda n: (
            setattr(n, "stability", min(100.0, n.stability + 20)),
            setattr(n.technology, "industry", min(100.0, n.technology.industry + 10.0)),
        ),
        {"Research Complex"},
    ),
    "Resilience Program": ProjectSpec(
        130.0,
        lambda n: setattr(n, "resilience", min(100.0, n.resilience + 30.0)),
    ),
    "Orbital Shipyard": ProjectSpec(
        150.0,
        lambda n: (
            setattr(n, "military", n.military + 30),
            setattr(n, "infrastructure", n.infrastructure + 10),
        ),
        {"Orbital Defense Grid"},
    ),
}

# Maintains deterministic order for project indexing
PROJECT_NAMES: List[str] = list(PROJECT_CATALOG.keys())

# Resource costs for constructing various assets
RESOURCE_COSTS: Dict[str, Dict[str, float]] = load_json("resource_costs")
