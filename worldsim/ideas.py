from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import List

from .culture import Culture
from .meta_ga import RewardGA


N_CULTURE_TRAITS = len(Culture.random().asdict())


@dataclass(slots=True)
class Idea:
    """Cultural meme that influences traits via a small GA."""

    name: str
    followers: int = 0
    ga: RewardGA = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.ga = RewardGA(N_CULTURE_TRAITS)

    def influence(self, culture: Culture, total_population: int) -> Culture:
        """Return modified culture based on follower ratio."""
        ratio = self.followers / max(1, total_population)
        result = culture.copy()
        weights: List[float] = self.ga.weights
        for idx, trait in enumerate(result.__dataclass_fields__):
            val = getattr(result, trait)
            delta = weights[idx % len(weights)] * ratio
            setattr(result, trait, max(0.0, min(1.0, val + delta)))
        return result

    def step(self, new_followers: int = 0) -> None:
        """Update follower count and GA fitness."""
        self.followers += new_followers
        self.ga.step()

    def evolve(self) -> None:
        """Evolve the GA controlling this idea."""
        self.ga.evolve()


@lru_cache(maxsize=None)
def get_idea(name: str) -> Idea:
    """Return an ``Idea`` instance for ``name`` using an internal cache."""
    return Idea(name)
