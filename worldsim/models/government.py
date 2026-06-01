"""Government archetype model."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, TYPE_CHECKING

from ..culture import ARCHETYPE_BONUSES, ArchetypeBonus

# Re-export moved symbols so existing ``from .government import X`` keeps working.
from .projects import NationalProject, ProjectSpec, PROJECT_CATALOG, PROJECT_NAMES
from .infrastructure import RESOURCE_COSTS

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
