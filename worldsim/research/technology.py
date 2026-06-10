"""Technology graph primitives and procedural tech generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Dict, List, Optional, Set, TYPE_CHECKING

import random

from .effects import _mult_bonus

if TYPE_CHECKING:
    from ..ai import ResearchAI
    from ..nations.nation import Nation

# Syllables used for procedural name generation of technologies
TECH_PREFIXES = [
    "Hyper",
    "Quantum",
    "Neo",
    "Solar",
    "Cryo",
    "Bio",
    "Cyber",
    "Nano",
    "Geo",
]

TECH_SUFFIXES = [
    "dynamics",
    "engineering",
    "systems",
    "theory",
    "design",
    "protocols",
    "network",
    "craft",
]

_TECH_COUNTER = count()


def generate_tech_name() -> str:
    """Return a unique technology name."""
    idx = next(_TECH_COUNTER)
    return f"{random.choice(TECH_PREFIXES)} {random.choice(TECH_SUFFIXES)} {idx}".title()


def generate_random_technology(tree: "TechnologyTree", nation: "Nation") -> "TechnologyNode":
    """Create a single random technology node with a unique name."""
    effects = [
        ("mine_output", 1.1),
        ("city_output", 1.1),
        ("division_power", 1.1),
        ("factory_output", 1.1),
        ("economy_mult", 1.05),
    ]
    key, factor = random.choice(effects)

    def _effect(n: "Nation", k=key, f=factor) -> None:
        _mult_bonus(n, k, f, cap=5.0 if k != "economy_mult" else 3.0)

    name = generate_tech_name()
    base_cost = random.randint(40, 90)
    cost = base_cost + len(tree.nodes)
    return TechnologyNode(name, cost, effect=_effect)


def add_random_technologies(tree: "TechnologyTree", nation: "Nation", count: int = 3) -> None:
    """Add ``count`` random technology nodes to ``tree``."""
    for _ in range(count):
        tree.add_node(generate_random_technology(tree, nation))


# ---------------------------------------------------------------------------
# Coarse technology levels (unchanged — consumed by events / projects / war)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Technology:
    """More detailed technology levels."""

    science: float = 10.0
    military: float = 10.0
    industry: float = 10.0

    def advance(self, economy: float, research_bonus: float = 0.0) -> None:
        """Advance technology based on ``economy`` and optional lab output."""
        # Basic gain from economic strength
        gain = economy / 3000
        # Additional scaling from research investment
        gain += research_bonus / 500
        self.science = min(100.0, self.science + gain)
        self.military = min(100.0, self.military + gain / 2)
        self.industry = min(100.0, self.industry + gain)

    @property
    def overall(self) -> float:
        return (self.science + self.military + self.industry) / 3


# ---------------------------------------------------------------------------
# Tech graph primitives
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TechnologyNode:
    """Node in a technology graph.

    ``effect`` is a first-class callable invoked once when the node is
    unlocked; it mutates ``nation.tech_bonuses`` (or the nation directly).
    """

    name: str
    cost: float
    prerequisites: Set[str] = field(default_factory=set)
    effect: Optional[callable] = None


@dataclass(slots=True)
class TechnologyTree:
    """A single subsystem's node graph plus its unlocked set and point pool.

    The class is deliberately a thin container.  Availability can be resolved
    against an *external* unlocked set so a subsystem graph can depend on
    technologies unlocked in a sibling subsystem (cross-subsystem
    prerequisites).  Allocation, AI selection and procedural growth all live on
    :class:`TechnologySubsystem` / :class:`ResearchDirector`.
    """

    nodes: Dict[str, TechnologyNode] = field(default_factory=dict)
    unlocked: Set[str] = field(default_factory=set)
    research_points: float = 0.0

    def add_node(self, node: TechnologyNode) -> None:
        self.nodes[node.name] = node

    def available(self, extra_unlocked: Optional[Set[str]] = None) -> List[TechnologyNode]:
        """Return researchable nodes.

        A node is researchable when it is not yet unlocked and all of its
        prerequisites are satisfied by the union of this tree's unlocked set
        and ``extra_unlocked`` (the rest of the nation's research).
        """
        if extra_unlocked is None:
            visible = self.unlocked
        else:
            visible = self.unlocked | extra_unlocked
        return [
            n
            for n in self.nodes.values()
            if n.name not in self.unlocked and n.prerequisites <= visible
        ]

    def research(self, points: float, nation: "Nation", ai: Optional["ResearchAI"] = None) -> None:
        """Spend ``points`` greedily on this tree in isolation.

        Retained for backwards compatibility / standalone use; the simulation
        drives research through :class:`ResearchDirector` instead.  Selection
        falls back to a cheapest-first heuristic — no AI state vector is
        fabricated here.
        """
        self.research_points += points
        options = self.available()
        while options and self.research_points >= min(n.cost for n in options):
            choice = min(options, key=lambda n: n.cost)
            self.research_points -= choice.cost
            self.unlocked.add(choice.name)
            if choice.effect:
                choice.effect(nation)
            options = self.available()

