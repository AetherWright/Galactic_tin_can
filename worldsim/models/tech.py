"""Three-subsystem technology architecture.

This module replaces the single flat :class:`TechnologyTree` with a layered
research model that mirrors the C# era design:

* Three :class:`TechnologySubsystem` instances — :class:`PhysicsSubsystem`,
  :class:`EngineeringSubsystem` and :class:`BiologySubsystem`.  Each owns its
  own named technology graph, its own :class:`~worldsim.ai.ResearchAI` and its
  own research-point pool.
* A :class:`ResearchDirector` that receives the nation's total research points
  each tick, allocates them between the subsystems and is aware of
  cross-subsystem prerequisites.

Design notes
------------
* **Output interface unchanged.**  Technologies still publish their effects
  through ``nation.tech_bonuses`` exactly as before — every effect is a
  first-class callable stored on the :class:`TechnologyNode`.
* **FTL stays in ftl.py.**  :func:`~worldsim.models.ftl.add_ftl_tech_nodes` is
  called against the engineering subsystem's graph; the drive prerequisites
  ("Industrial Automation", "Atomic Engineering", "Nuclear Weapons") live in
  the engineering / physics subsystems, so the FTL chain is gated across both
  via the cross-subsystem prerequisite mechanism.  ftl.py is not modified.
* **SpacecraftTechnology stays separate.**  As in the C# version, ship-class
  technology (``ShipTemplate`` / ``SHIP_TEMPLATES`` in ``fleet.py``) is *not*
  merged into the subsystems.  Only FTL propulsion — which is engineering —
  lives here.
* **Procedural generation is gated.**  Random technologies are only grown once
  *all* of a subsystem's base technologies are unlocked, never continuously
  while the hand-authored chain is still in progress.
* The ``nation.tech_tree`` attribute now holds the :class:`ResearchDirector`.
  It still quacks like the old tree (``.unlocked``, ``.nodes``, ``.research``)
  so the rest of the simulation needs no changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Dict, List, Optional, Set

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


def _add_bonus(n: "Nation", key: str, amount: float) -> None:
    n.tech_bonuses[key] = n.tech_bonuses.get(key, 0.0) + amount


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


# ---------------------------------------------------------------------------
# Subsystems
# ---------------------------------------------------------------------------

class TechnologySubsystem:
    """One research domain: a node graph, a :class:`ResearchAI` and a pool.

    Subclasses populate :attr:`tree` with their hand-authored named
    technologies in :meth:`build_nodes`.  Cross-subsystem prerequisites are
    resolved against the owning :class:`ResearchDirector`'s global unlocked set
    via :meth:`global_unlocked`.
    """

    #: Human-readable subsystem name (also used as the AI table role suffix).
    name: str = "subsystem"
    #: Input dimensionality for this subsystem's ResearchAI; overridden per subclass.
    _n_inputs: int = 4

    def __init__(self) -> None:
        self.tree = TechnologyTree()
        self.ai: Optional[ResearchAI] = None
        self.director: Optional["ResearchDirector"] = None
        #: Names of the hand-authored (non-procedural) technologies.  Populated
        #: by :meth:`finalize_bases` after the director has injected any extra
        #: nodes (e.g. FTL).  Procedural growth is gated on all of these being
        #: unlocked.
        self.base_names: Set[str] = set()
        #: Whether this subsystem may grow procedural nodes once its base graph
        #: is fully unlocked.
        self.allow_procedural: bool = True

    # -- views -------------------------------------------------------------
    @property
    def unlocked(self) -> Set[str]:
        return self.tree.unlocked

    @property
    def nodes(self) -> Dict[str, TechnologyNode]:
        return self.tree.nodes

    def global_unlocked(self) -> Set[str]:
        """Return everything unlocked across the whole nation."""
        if self.director is not None:
            return self.director.unlocked
        return self.tree.unlocked

    def available(self) -> List[TechnologyNode]:
        return self.tree.available(self.global_unlocked())

    # -- construction hooks ------------------------------------------------
    def build_nodes(self, nation: "Nation") -> None:
        """Populate :attr:`tree` with named technologies.  Override me."""
        raise NotImplementedError

    def finalize_bases(self) -> None:
        """Record the full base graph (called after all injection is done)."""
        self.base_names = set(self.tree.nodes)

    def attach_ai(self) -> None:
        """Create this subsystem's :class:`ResearchAI` using :attr:`_n_inputs`."""
        n_actions = len(self.tree.nodes)
        if n_actions <= 0:
            self.ai = None
            return
        # # TODO: Aether — hidden-layer topology per subsystem; the ResearchAI
        # # default ``(10, 8, 6, 4, 3, 4, 6, 8, 10)`` is a placeholder.
        self.ai = ResearchAI(n_actions, n_inputs=self._n_inputs)

    # -- per-tick research -------------------------------------------------
    def _build_state(self, nation: "Nation") -> Optional[List[float]]:
        """Return the AI state vector for technology selection.

        Subclasses with a defined state vector override this.  Returning
        ``None`` makes :meth:`research` fall back to cheapest-first heuristic.

        # TODO: Aether — define state-vector contents for biology (and any
        # future subsystem) once their research domain is populated.
        """
        return None

    def research(self, points: float, nation: "Nation") -> None:
        """Add ``points`` to the pool and unlock as many techs as affordable."""
        self.tree.research_points += points
        options = self.available()
        while options and self.tree.research_points >= min(n.cost for n in options):
            state = self._build_state(nation)
            if self.ai is not None and state is not None:
                idx = self.ai.choose_action(state) % len(options)
                choice = options[idx]
            else:
                idx = None
                choice = min(options, key=lambda n: n.cost)
            if self.tree.research_points < choice.cost:
                break
            self.tree.research_points -= choice.cost
            self.tree.unlocked.add(choice.name)
            if choice.effect:
                choice.effect(nation)
            self._maybe_grow(nation)
            if self.ai is not None and state is not None and idx is not None:
                new_state = self._build_state(nation)
                reward = nation.compute_reward("research", state, new_state)
                self.ai.train(state, idx, reward, new_state)
            options = self.available()

    def _maybe_grow(self, nation: "Nation") -> None:
        """Grow a procedural node, but only after the base graph is complete."""
        if not self.allow_procedural or not self.base_names:
            return
        if not self.base_names <= self.tree.unlocked:
            return
        add_random_technologies(self.tree, nation, 1)
        if self.ai is not None:
            self.ai.n_actions = len(self.tree.nodes)


# ---------------------------------------------------------------------------
# Effect helpers shared by the hand-authored civilization chain
# ---------------------------------------------------------------------------
# Effects remain first-class callables on the nodes and publish exclusively
# through ``nation.tech_bonuses`` (the unchanged output interface).

def _city_output(n: "Nation") -> None:
    n.tech_bonuses["city_output"] = n.tech_bonuses.get("city_output", 1.0) * 1.1


def _admin(n: "Nation") -> None:
    n.tech_bonuses["economy_mult"] = n.tech_bonuses.get("economy_mult", 1.0) * 1.1


def _math_admin(n: "Nation") -> None:
    n.tech_bonuses["economy_mult"] = n.tech_bonuses.get("economy_mult", 1.0) * 1.05


def _factory(n: "Nation") -> None:
    n.tech_bonuses["factory_output"] = n.tech_bonuses.get("factory_output", 1.0) * 1.1


def _factory_major(n: "Nation") -> None:
    n.tech_bonuses["factory_output"] = n.tech_bonuses.get("factory_output", 1.0) * 1.2


def _mine_output(n: "Nation") -> None:
    n.tech_bonuses["mine_output"] = n.tech_bonuses.get("mine_output", 1.0) * 1.1


def _atomic(n: "Nation") -> None:
    n.tech_bonuses["nuclear_prod"] = 1.0


def _nuclear(n: "Nation") -> None:
    n.tech_bonuses["nuclear_power"] = 1.0


# -- Biology: Ecology track -------------------------------------------------
def _botany(n: "Nation") -> None:
    _mult_bonus(n, "city_output", 1.05)
    _add_bonus(n, "plague_resist", 0.05)


def _ecosystem_mapping(n: "Nation") -> None:
    # Signals terraforming readiness and improves colony viability.
    n.tech_bonuses["terraform_readiness"] = 1.0
    _mult_bonus(n, "colony_viability", 1.1)


def _synthetic_ecology(n: "Nation") -> None:
    _mult_bonus(n, "habitability", 1.15)


def _terraforming(n: "Nation") -> None:
    # Flag consumed by colonization to unlock new colony types.
    n.tech_bonuses["terraforming"] = 1.0


# -- Biology: Medicine track ------------------------------------------------
def _herbalism(n: "Nation") -> None:
    _add_bonus(n, "plague_resist", 0.1)


def _medicine(n: "Nation") -> None:
    _add_bonus(n, "plague_resist", 0.2)
    _mult_bonus(n, "hospital_effectiveness", 1.1)


def _pharmacology(n: "Nation") -> None:
    _add_bonus(n, "plague_resist", 0.3)
    _mult_bonus(n, "pop_growth", 1.1)


def _advanced_medicine(n: "Nation") -> None:
    _add_bonus(n, "plague_resist", 0.4)


def _genetic_medicine(n: "Nation") -> None:
    _add_bonus(n, "plague_resist", 0.5)
    _mult_bonus(n, "resilience_bonus", 1.1)


# -- Biology: Genetics track ------------------------------------------------
def _cell_biology(n: "Nation") -> None:
    _mult_bonus(n, "research_mult", 1.1)
    _mult_bonus(n, "lab_effectiveness", 1.1)


def _genetic_engineering(n: "Nation") -> None:
    _mult_bonus(n, "pop_cap", 1.1)
    _mult_bonus(n, "mutation_rate", 1.1)


def _synthetic_biology(n: "Nation") -> None:
    _mult_bonus(n, "food_output", 1.15)


def _directed_evolution(n: "Nation") -> None:
    _mult_bonus(n, "adaptive_colonization", 1.1)


# ---------------------------------------------------------------------------
# State-vector helpers for subsystem ResearchAIs
# ---------------------------------------------------------------------------

def _cross_demand_pressure(nation: "Nation") -> float:
    """Fraction of engineering nodes blocked specifically by unmet physics prereqs."""
    director = nation.tech_tree
    if not hasattr(director, "physics"):
        return 0.0
    physics_names: Set[str] = set(director.physics.nodes)
    global_unlocked: Set[str] = director.unlocked
    eng_unlocked: Set[str] = director.engineering.unlocked
    blocked = sum(
        1
        for node in director.engineering.nodes.values()
        if node.name not in eng_unlocked
        and (node.prerequisites - global_unlocked) & physics_names
    )
    return blocked / max(1, len(director.engineering.nodes))


def _ftl_unlock_ratio(nation: "Nation") -> float:
    """Fraction of the FTL tier ladder that has been unlocked."""
    from .ftl import FTL_TIER_ORDER
    unlocked = nation.tech_tree.unlocked
    return sum(1 for name in FTL_TIER_ORDER if name in unlocked) / len(FTL_TIER_ORDER)


class PhysicsSubsystem(TechnologySubsystem):
    """Fundamental science: the theoretical half of the civilization chain.

    Holds the abstract advances that gate applied engineering work.  The chain
    zig-zags across physics and engineering, demonstrating cross-subsystem
    prerequisites (e.g. "Mathematics" here requires "Writing" from engineering).
    """

    name = "physics"
    _n_inputs = 8

    def _build_state(self, nation: "Nation") -> Optional[List[float]]:
        total_nodes = max(
            1,
            sum(len(s.nodes) for s in self.director.subsystems)
            if self.director is not None
            else len(self.nodes),
        )
        return [
            nation.technology.science / 100.0,           # 0: current science level
            nation.technology.industry / 100.0,          # 1: industry (cross-subsystem signal)
            len(self.unlocked) / max(1, len(self.nodes)), # 2: physics completion ratio
            len(self.available()) / max(1, len(self.nodes)), # 3: available-node ratio
            len(self.global_unlocked()) / total_nodes,   # 4: global research progress
            nation.economy / 10000.0,                    # 5: economy (normalised)
            float(len(nation.labs)) / 10.0,              # 6: research infrastructure
            _cross_demand_pressure(nation),              # 7: eng nodes blocked by physics prereqs
        ]

    def build_nodes(self, nation: "Nation") -> None:
        t = self.tree
        # Civilization progression chain (physics portion).  Prerequisites that
        # name engineering technologies are resolved against the global
        # unlocked set by the director.
        t.add_node(TechnologyNode("Mathematics", 60, {"Writing"}, _math_admin))
        t.add_node(TechnologyNode("Material Science", 80, {"Engineering"}, _mine_output))
        t.add_node(TechnologyNode("Atomic Engineering", 120, {"Industrialization"}, _atomic))
        # # TODO: Aether — additional physics technologies and their effects
        # # beyond the civilization progression chain (e.g. field theory,
        # # particle physics, exotic-matter research).


class EngineeringSubsystem(TechnologySubsystem):
    """Applied construction: the practical half of the civilization chain.

    Also the home of FTL propulsion, injected by the director through
    :func:`~worldsim.models.ftl.add_ftl_tech_nodes`.
    """

    name = "engineering"
    _n_inputs = 12

    def _build_state(self, nation: "Nation") -> Optional[List[float]]:
        return [
            nation.technology.industry / 100.0,              # 0: industry level
            nation.technology.military / 100.0,              # 1: military level
            len(self.unlocked) / max(1, len(self.nodes)),    # 2: engineering completion ratio
            len(self.available()) / max(1, len(self.nodes)), # 3: available-node ratio
            nation.economy / 10000.0,                        # 4: economy (normalised)
            float(len(nation.factories)) / 10.0,             # 5: factory infrastructure
            float(len(nation.mines)) / 10.0,                 # 6: mining infrastructure
            float(len(nation.at_war)) / 10.0,                # 7: war pressure
            float(nation.nuclear_stockpile) / 100.0,         # 8: nuclear development
            _ftl_unlock_ratio(nation),                       # 9: FTL tier progress
            float(len(nation.cities)) / 50.0,                # 10: city count
            float(nation.star_count) / 10.0,                 # 11: stellar footprint
        ]

    def build_nodes(self, nation: "Nation") -> None:
        t = self.tree
        # Civilization progression chain (engineering portion).
        t.add_node(TechnologyNode("Agriculture", 50, effect=_city_output))
        t.add_node(TechnologyNode("Writing", 55, {"Agriculture"}, _admin))
        t.add_node(TechnologyNode("Engineering", 70, {"Mathematics"}, _factory))
        t.add_node(TechnologyNode("Industrialization", 90, {"Material Science"}, _factory_major))
        # Required by ftl.py prerequisites + war/events code.  Kept under their
        # original names so the FTL chain and "Nuclear Weapons" / "Atomic
        # Engineering" checks elsewhere continue to resolve.
        t.add_node(TechnologyNode("Industrial Automation", 80, {"Industrialization"}, _factory))
        t.add_node(TechnologyNode("Nuclear Weapons", 150, {"Atomic Engineering"}, _nuclear))
        # # TODO: Aether — additional engineering technologies and their effects
        # # beyond the civilization progression chain (e.g. military tactics,
        # # weaponry, armored doctrine, shipbuilding for ports).


class BiologySubsystem(TechnologySubsystem):
    """Life sciences: a three-track tree of ecology, medicine and genetics.

    The tracks interleave (e.g. genetics' "Genetic Engineering" gates ecology's
    "Synthetic Ecology") and one node ("Terraforming") depends on engineering's
    "Industrialization", exercising both cross-track and cross-subsystem
    prerequisites.  All tracks root at "Botany".

    # TODO: Aether — biology ResearchAI state vector.  Until ``_build_state``
    # is defined (it returns ``None`` via the base class), node selection uses
    # the cheapest-first heuristic and the biology AI stays untrained.
    """

    name = "biology"
    # # TODO: Aether — biology-specific _n_inputs once its state vector exists;
    # # currently the AI is constructed but never invoked (heuristic selection).

    def build_nodes(self, nation: "Nation") -> None:
        t = self.tree
        # Ecology track
        t.add_node(TechnologyNode("Botany", 50, effect=_botany))
        t.add_node(TechnologyNode("Ecosystem Mapping", 70, {"Botany"}, _ecosystem_mapping))
        t.add_node(TechnologyNode(
            "Synthetic Ecology", 120,
            {"Ecosystem Mapping", "Genetic Engineering"}, _synthetic_ecology,
        ))
        t.add_node(TechnologyNode(
            "Terraforming", 160,
            {"Synthetic Ecology", "Industrialization"}, _terraforming,
        ))
        # Medicine track
        t.add_node(TechnologyNode("Herbalism", 60, {"Botany"}, _herbalism))
        t.add_node(TechnologyNode("Medicine", 80, {"Herbalism"}, _medicine))
        t.add_node(TechnologyNode(
            "Pharmacology", 100, {"Medicine", "Ecosystem Mapping"}, _pharmacology,
        ))
        t.add_node(TechnologyNode("Advanced Medicine", 110, {"Pharmacology"}, _advanced_medicine))
        t.add_node(TechnologyNode(
            "Genetic Medicine", 150,
            {"Advanced Medicine", "Genetic Engineering"}, _genetic_medicine,
        ))
        # Genetics track
        t.add_node(TechnologyNode("Cell Biology", 90, {"Medicine"}, _cell_biology))
        t.add_node(TechnologyNode(
            "Genetic Engineering", 130,
            {"Cell Biology", "Pharmacology"}, _genetic_engineering,
        ))
        t.add_node(TechnologyNode("Synthetic Biology", 140, {"Genetic Engineering"}, _synthetic_biology))
        t.add_node(TechnologyNode(
            "Directed Evolution", 150,
            {"Synthetic Biology", "Ecosystem Mapping"}, _directed_evolution,
        ))


# ---------------------------------------------------------------------------
# Research director
# ---------------------------------------------------------------------------

class ResearchDirector:
    """Owns the three subsystems and allocates research points between them.

    Exposes the legacy ``TechnologyTree`` surface (``unlocked`` / ``nodes`` /
    ``research``) so the rest of the simulation can treat ``nation.tech_tree``
    unchanged.
    """

    def __init__(self, nation: "Nation") -> None:
        self.physics = PhysicsSubsystem()
        self.engineering = EngineeringSubsystem()
        self.biology = BiologySubsystem()
        self.subsystems: List[TechnologySubsystem] = [
            self.physics,
            self.engineering,
            self.biology,
        ]
        for sub in self.subsystems:
            sub.director = self
            sub.build_nodes(nation)

        # Inject FTL drives into the engineering subsystem's graph.  The drive
        # prerequisites span engineering ("Industrial Automation", "Nuclear
        # Weapons") and physics ("Atomic Engineering"), so the FTL chain is
        # gated across both subsystems through cross-subsystem prerequisites.
        # ftl.py is not modified.  Spacecraft / ship-class technology remains
        # separate (see fleet.py) and is deliberately NOT added here.
        from .ftl import add_ftl_tech_nodes
        add_ftl_tech_nodes(self.engineering.tree, nation)

        # Lock in the base graphs (incl. injected FTL nodes) and build AIs.
        for sub in self.subsystems:
            sub.finalize_bases()
            sub.attach_ai()

        #: Director-level allocation policy.
        # # TODO: Aether — director allocation strategy.  Optionally drive this
        # # with a dedicated allocation AI (a ResearchAI over the three
        # # subsystems, or a learned policy keyed on cross-subsystem demand).
        # # The default below is a transparent heuristic.
        self.allocation_ai: Optional[ResearchAI] = None

        #: Cross-subsystem dependency graph beyond the civilization chain.
        # # TODO: Aether — declare which physics technologies gate which
        # # engineering (or biology) technologies, beyond the hand-wired
        # # prerequisites already on the civilization-chain nodes.  Format e.g.
        # # ``{"Field Theory": {"Antimatter Containment"}}``.  These would be
        # # merged into the relevant node prerequisites at build time.
        self.cross_subsystem_prereqs: Dict[str, Set[str]] = {}

        self.total_points: float = 0.0

    # -- legacy TechnologyTree surface ------------------------------------
    @property
    def unlocked(self) -> Set[str]:
        """Union of every subsystem's unlocked technologies."""
        out: Set[str] = set()
        for sub in self.subsystems:
            out |= sub.tree.unlocked
        return out

    @property
    def nodes(self) -> Dict[str, TechnologyNode]:
        """Merged view of every subsystem's nodes."""
        out: Dict[str, TechnologyNode] = {}
        for sub in self.subsystems:
            out.update(sub.tree.nodes)
        return out

    # -- allocation --------------------------------------------------------
    def research(
        self, points: float, nation: "Nation", ai: Optional["ResearchAI"] = None
    ) -> None:
        """Allocate ``points`` across the subsystems and run their research.

        ``ai`` is accepted for signature compatibility with the old
        ``TechnologyTree.research`` call (``nation.research_ai``) but is not
        used for allocation by default — see the ``allocation_ai`` TODO.
        """
        self.total_points += points
        if points <= 0:
            return

        # # TODO: Aether — replace this heuristic with the real allocation
        # # strategy (priority weighting, demand-aware splitting, or
        # # allocation_ai-driven choice).  For now: distribute the tick's points
        # # across subsystems weighted by how many technologies each can
        # # currently research, so subsystems with more open work receive
        # # proportionally more.  Idle subsystems (e.g. an unpopulated biology
        # # graph) get nothing while others have researchable techs.
        researchable = [(sub, sub.available()) for sub in self.subsystems]
        active = [(sub, avail) for sub, avail in researchable if avail]
        if not active:
            # Everything currently researchable is done; keep feeding all
            # subsystems evenly so their pools build and procedural growth can
            # proceed.
            share = points / len(self.subsystems)
            for sub in self.subsystems:
                sub.research(share, nation)
            return

        total = sum(len(avail) for _, avail in active)
        for sub, avail in active:
            sub.research(points * len(avail) / total, nation)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def setup_default_tech_tree(nation: "Nation") -> None:
    """Install a :class:`ResearchDirector` on ``nation`` as its tech tree."""
    nation.tech_tree = ResearchDirector(nation)
