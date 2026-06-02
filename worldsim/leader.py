from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import TYPE_CHECKING, List

from .culture import Culture
from .meta_ga import RewardGA

if TYPE_CHECKING:  # pragma: no cover
    from .models import Nation


@dataclass
class Leader:
    """Represents a nation's leader with personal cultural biases."""

    name: str
    culture: Culture


@dataclass
class LeaderModel:
    """Generator producing leaders influenced by national ideas."""

    ga: RewardGA = field(default_factory=lambda: RewardGA(len(Culture.random().asdict())))

    def _ensure_culture(self, obj) -> Culture:
        if isinstance(obj, Culture):
            return obj
        if isinstance(obj, dict):
            c = Culture.random()
            for trait in c.__dataclass_fields__:
                if trait in obj:
                    setattr(c, trait, float(obj[trait]))
            return c
        return Culture.random()

    def generate(self, nation: "Nation") -> Leader:
        """Return a new Leader with culture modified by nation's idea strength."""
        base = self._ensure_culture(nation.culture_with_ideas())
        total_followers = sum(i.followers for i in nation.ideas.values())
        ratio = total_followers / max(1, nation.population)
        result = base.copy()
        weights: List[float] = self.ga.weights
        for idx, trait in enumerate(result.__dataclass_fields__):
            val = getattr(result, trait)
            delta = weights[idx % len(weights)] * ratio
            setattr(result, trait, max(0.0, min(1.0, val + delta)))
        name = f"Leader-{random.randint(0, 9999)}"
        return Leader(name, result)

    def step(self) -> None:
        self.ga.step(1)

    def evolve(self) -> None:
        self.ga.evolve()
