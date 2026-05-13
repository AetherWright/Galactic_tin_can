from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Dict, Set, List, Optional

import random

from ..ai import ResearchAI
from .war import DivisionTemplate

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

def _mult_bonus(n: "Nation", key: str, factor: float, cap: float = 5.0) -> None:
    n.tech_bonuses[key] = min(n.tech_bonuses.get(key, 1.0) * factor, cap)

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


@dataclass(slots=True)
class TechnologyNode:
    """Node in a technology tree."""

    name: str
    cost: float
    prerequisites: Set[str] = field(default_factory=set)
    effect: Optional[callable] = None


@dataclass(slots=True)
class TechnologyTree:
    """Collection of technologies to research."""

    nodes: Dict[str, TechnologyNode] = field(default_factory=dict)
    unlocked: Set[str] = field(default_factory=set)
    research_points: float = 0.0

    def add_node(self, node: TechnologyNode) -> None:
        self.nodes[node.name] = node

    def available(self) -> List[TechnologyNode]:
        return [n for n in self.nodes.values() if n.name not in self.unlocked and n.prerequisites <= self.unlocked]

    def research(self, points: float, nation: "Nation", ai: Optional["ResearchAI"] = None) -> None:
        self.research_points += points
        options = self.available()
        while options and self.research_points >= min(n.cost for n in options):
            if ai:
                state = [
                    nation.technology.science,
                    nation.technology.military,
                    nation.technology.industry,
                    len(self.unlocked),
                ]
                idx = ai.choose_action(state)
                idx = idx % len(options)
                choice = options[idx]
            else:
                choice = options[0]
            if self.research_points < choice.cost:
                break
            self.research_points -= choice.cost
            self.unlocked.add(choice.name)
            if choice.effect:
                choice.effect(nation)
            # Dynamically grow the tech tree when research completes
            add_random_technologies(self, nation, 1)
            if nation.research_ai:
                nation.research_ai.n_actions = len(self.nodes)
            options = self.available()
            if ai:
                new_state = [
                    nation.technology.science,
                    nation.technology.military,
                    nation.technology.industry,
                    len(self.unlocked),
                ]
                reward = nation.compute_reward("research", state, new_state)
                ai.train(state, idx, reward, new_state)
def setup_default_tech_tree(nation: "Nation") -> None:
    """Populate a basic tech tree for ``nation``."""

    tree = TechnologyTree()

    def mining_bonus(n: "Nation") -> None:
        n.tech_bonuses["mine_output"] = n.tech_bonuses.get("mine_output", 1.0) * 1.1

    def farming_bonus(n: "Nation") -> None:
        n.tech_bonuses["city_output"] = n.tech_bonuses.get("city_output", 1.0) * 1.1

    def medical_bonus(n: "Nation") -> None:
        n.tech_bonuses["plague_resist"] = n.tech_bonuses.get("plague_resist", 0.0) + 0.2

    def tactics_bonus(n: "Nation") -> None:
        _mult_bonus(n, "division_power", 1.1, cap=5.0)

    def weapons_bonus(n: "Nation") -> None:
        _mult_bonus(n, "division_power", 1.2, cap=5.0)

    def ship_bonus(n: "Nation") -> None:
        n.tech_bonuses["port_bonus"] = n.tech_bonuses.get("port_bonus", 1.0) * 1.1

    def admin_bonus(n: "Nation") -> None:
        n.tech_bonuses["economy_mult"] = n.tech_bonuses.get("economy_mult", 1.0) * 1.1

    def industry_bonus(n: "Nation") -> None:
        n.tech_bonuses["factory_output"] = n.tech_bonuses.get("factory_output", 1.0) * 1.1

    def armored_unit(n: "Nation") -> None:
        n.division_templates["Armored"] = DivisionTemplate("Armored", 600, 1.5, 1.0)

    def atomic_bonus(n: "Nation") -> None:
        n.tech_bonuses["nuclear_prod"] = 1.0

    def nuclear_weapon(n: "Nation") -> None:
        n.tech_bonuses["nuclear_power"] = 1.0

    tree.add_node(TechnologyNode("Improved Mining", 50, effect=mining_bonus))
    tree.add_node(TechnologyNode("Advanced Agriculture", 50, effect=farming_bonus))
    tree.add_node(TechnologyNode("Medical Knowledge", 60, effect=medical_bonus))
    tree.add_node(TechnologyNode("Military Tactics", 70, {"Medical Knowledge"}, tactics_bonus))
    tree.add_node(TechnologyNode("Shipbuilding", 70, effect=ship_bonus))
    tree.add_node(TechnologyNode("Efficient Administration", 70, effect=admin_bonus))
    tree.add_node(TechnologyNode("Improved Weaponry", 80, {"Military Tactics"}, weapons_bonus))
    tree.add_node(TechnologyNode("Industrial Automation", 80, {"Improved Mining"}, industry_bonus))
    tree.add_node(TechnologyNode("Armored Warfare", 100, {"Improved Weaponry"}, armored_unit))
    tree.add_node(TechnologyNode("Atomic Engineering", 120, {"Industrial Automation"}, atomic_bonus))
    tree.add_node(TechnologyNode("Nuclear Weapons", 150, {"Atomic Engineering", "Improved Weaponry"}, nuclear_weapon))

    # Add a few randomised technologies so each nation differs slightly
    add_random_technologies(tree, nation)

    # Inject FTL drive nodes so the full interstellar tech chain is available
    from .ftl import add_ftl_tech_nodes
    add_ftl_tech_nodes(tree, nation)

    nation.tech_tree = tree
