"""Simple dynamic goal management for nations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence
import random


@dataclass
class Goal:
    """Represents a measurable objective."""

    name: str
    metric: str
    target: float
    priority: float = 1.0
    completed: bool = False

    start_value: float | None = None

    def _resolve(self, nation) -> float:
        if self.metric == "economy":
            return float(getattr(nation, "economy_linear", getattr(nation, "economy", 0.0)))
        obj = nation
        for part in self.metric.split("."):
            obj = getattr(obj, part, 0.0)
        return float(obj)

    def check(self, nation) -> bool:
        """Update completion status using *nation*'s state."""
        value = self._resolve(nation)
        if self.start_value is None:
            self.start_value = value
        if value >= self.target:
            self.completed = True
        return self.completed

    def progress(self, nation) -> float:
        """Return completion ratio from 0.0 to 1.0."""
        if self.start_value is None:
            self.start_value = self._resolve(nation)
        span = self.target - self.start_value
        if span <= 0:
            return 1.0
        value = self._resolve(nation)
        return max(0.0, min((value - self.start_value) / span, 1.0))


@dataclass
class GoalTemplate:
    """Pattern used to create a new goal for a nation."""

    name: str
    metric: str
    multiplier: float = 1.1
    offset: float = 0.0

    def create(self, nation) -> Goal:
        current = Goal("tmp", self.metric, 0.0)._resolve(nation)
        target = current * self.multiplier + self.offset
        return Goal(self.name.format(target=target), self.metric, target)


class GoalManager:
    """Container and helper for multiple goals."""

    def __init__(self) -> None:
        self.goals: List[Goal] = []

    def spawn_goals(
        self, nation, count: int = 1, templates: Sequence[GoalTemplate] | None = None
    ) -> None:
        templates = list(templates or DEFAULT_TEMPLATES)
        for _ in range(count):
            tmpl = random.choice(templates)
            self.add(tmpl.create(nation))

    def add(self, goal: Goal) -> None:
        self.goals.append(goal)

    def active(self) -> List[Goal]:
        return [g for g in self.goals if not g.completed]

    def update(self, nation) -> None:
        for g in self.goals:
            if not g.completed:
                g.check(nation)


DEFAULT_TEMPLATES: List[GoalTemplate] = [
    GoalTemplate("Grow economy to {target:.1f}", "economy", 1.1),
    GoalTemplate("Boost population to {target:.0f}", "population", 1.1),
    GoalTemplate("Raise stability to {target:.1f}", "stability", 1.05),
    GoalTemplate("Expand territory to {target:.0f}", "territory", 1.1),
    GoalTemplate("Advance science to {target:.1f}", "technology.science", 1.05),
]

