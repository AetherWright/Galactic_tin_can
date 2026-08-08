from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Callable
import random

@dataclass(slots=True)
class Culture:
    """Container holding several cultural traits.

    Additional traits can be added over time without changing the
    surrounding code as iteration uses ``__dataclass_fields__``.
    """

    military: float
    trade: float
    science: float
    diplomacy: float
    spirituality: float
    art: float
    imperialism: float
    industrialism: float

    @classmethod
    def random(cls) -> "Culture":
        """Return a randomly initialised culture."""
        return cls(*(random.random() for _ in range(8)))

    def asdict(self) -> Dict[str, float]:
        """Return dictionary representation."""
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def copy(self) -> "Culture":
        """Return a copy of this culture."""
        return Culture(**self.asdict())

    def mutate(self, rate: float = 0.05) -> None:
        """Jitter all traits by ``rate`` while clamping between 0 and 1."""
        for name in self.__dataclass_fields__:
            val = getattr(self, name)
            val += random.uniform(-rate, rate)
            setattr(self, name, max(0.0, min(1.0, val)))


@dataclass
class ArchetypeBonus:
    """Bonus profile for a single governance archetype at full expression."""
    
    # Flat additive bonuses
    stability_flat: float = 0.0
    military_flat: float = 0.0
    military_mult: float = 1.0
    diplomacy_mult: float = 1.0
    trade_mult: float = 1.0
    economy_mult: float = 1.0
    science_mult: float = 1.0
    art_mult: float = 1.0
    infrastructure_mult: float = 1.0
    plague_resist: float = 0.0
    ship_cost_mult: float = 1.0
    tribute_mult: float = 1.0
    annexation_mult: float = 1.0
    
    # Conditional/scaling modifiers
    stability_decay_at_peace: float = 0.0      # per century without wars
    stability_decay_no_expansion: float = 0.0  # per century without new stars/cities
    stability_scale_per_star: float = 0.0      # bonus per star owned
    economy_scale_with_science: bool = False    # Technocracy science-economy link
    economy_penalty_small: bool = False         # Empire penalty below 3 stars
    civil_war_rebel_strength: float = 1.0      # Democracy multiplier
    ally_trade_bonus: float = 0.0              # Socialism allied trade

ARCHETYPE_IDEALS = {
    "Technocracy":       [0.2, 0.3, 0.9, 0.5, 0.1, 0.3, 0.1, 0.8],
    "Military Junta":    [0.9, 0.2, 0.3, 0.2, 0.3, 0.2, 0.5, 0.5],
    "Democracy":         [0.3, 0.6, 0.6, 0.9, 0.4, 0.6, 0.2, 0.4],
    "Empire":            [0.7, 0.5, 0.4, 0.4, 0.3, 0.5, 0.9, 0.6],
    "Socialism":         [0.3, 0.5, 0.5, 0.7, 0.4, 0.5, 0.1, 0.7],
    "Fascism":           [0.8, 0.2, 0.3, 0.1, 0.2, 0.3, 0.8, 0.7],
    "Theocracy":         [0.4, 0.3, 0.2, 0.5, 0.9, 0.7, 0.3, 0.2],
    "Merchant Republic": [0.2, 0.9, 0.5, 0.8, 0.2, 0.5, 0.2, 0.7],
}

ARCHETYPE_BONUSES: Dict[str, ArchetypeBonus] = {
    "Technocracy": ArchetypeBonus(
        stability_flat=2.0,
        science_mult=1.0,  # handled by economy_scale_with_science
        economy_mult=1.05,
        infrastructure_mult=1.05,
        diplomacy_mult=0.95,
        art_mult=0.9,
        economy_scale_with_science=True,
    ),
    "Military Junta": ArchetypeBonus(
        military_flat=15.0,
        stability_flat=5.0,
        economy_mult=0.9,
        science_mult=0.95,
        stability_decay_at_peace=2.0,
    ),
    "Democracy": ArchetypeBonus(
        stability_flat=5.0,
        diplomacy_mult=1.15,
        trade_mult=1.10,
        science_mult=1.05,
        civil_war_rebel_strength=1.5,
    ),
    "Empire": ArchetypeBonus(
        military_mult=1.05,
        trade_mult=1.10,
        tribute_mult=1.20,
        annexation_mult=1.25,
        stability_scale_per_star=0.5,
        economy_penalty_small=True,
    ),
    "Socialism": ArchetypeBonus(
        stability_flat=3.0,
        diplomacy_mult=1.15,
        science_mult=1.05,
        economy_mult=1.0,
        military_mult=0.95,
        ally_trade_bonus=0.15,
    ),
    "Fascism": ArchetypeBonus(
        military_flat=20.0,
        stability_flat=-5.0,
        military_mult=1.4,
        economy_mult=1.0,
        trade_mult=0.85,
        diplomacy_mult=0.80,
        stability_decay_no_expansion=3.0,
    ),
    "Theocracy": ArchetypeBonus(
        stability_flat=8.0,
        plague_resist=0.30,
        art_mult=1.10,
        science_mult=0.85,
        diplomacy_mult=1.05,
    ),
    "Merchant Republic": ArchetypeBonus(
        trade_mult=1.25,
        diplomacy_mult=1.15,
        stability_flat=-3,
        economy_mult=1.15,
        art_mult=1.05,
        military_mult=0.80,
        ship_cost_mult=0.90,
    ),
}
