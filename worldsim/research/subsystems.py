"""Research subsystems: physics, engineering and biology domains.

Each subsystem owns its own named technology graph, its own
:class:`~worldsim.ai.ResearchAI` and its own research-point pool.
Cross-subsystem prerequisites are resolved against the owning
:class:`~worldsim.research.director.ResearchDirector`'s global unlocked set.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, TYPE_CHECKING

from ..ai import ResearchAI
from .effects import (
    _add_bonus,
    _admin,
    _advanced_medicine,
    _atomic,
    _botany,
    _cell_biology,
    _city_output,
    _directed_evolution,
    _ecosystem_mapping,
    _factory,
    _factory_major,
    _genetic_engineering,
    _genetic_medicine,
    _herbalism,
    _math_admin,
    _medicine,
    _mine_output,
    _mult_bonus,
    _nuclear,
    _pharmacology,
    _synthetic_biology,
    _synthetic_ecology,
    _terraforming,
)
from .technology import TechnologyNode, TechnologyTree, add_random_technologies

if TYPE_CHECKING:
    from ..nations.nation import Nation
    from .director import ResearchDirector


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
    from ..military.ftl import FTL_TIER_ORDER
    unlocked = nation.tech_tree.unlocked
    return sum(1 for name in FTL_TIER_ORDER if name in unlocked) / len(FTL_TIER_ORDER)


# Biology track membership — drives the per-track completion state features.
_ECOLOGY_TRACK = ("Botany", "Ecosystem Mapping", "Synthetic Ecology", "Terraforming")
_MEDICINE_TRACK = (
    "Herbalism", "Medicine", "Pharmacology", "Advanced Medicine", "Genetic Medicine",
)
_GENETICS_TRACK = (
    "Cell Biology", "Genetic Engineering", "Synthetic Biology", "Directed Evolution",
)


def _track_completion(subsystem: "TechnologySubsystem", track) -> float:
    unlocked = subsystem.unlocked
    return sum(1 for name in track if name in unlocked) / len(track)


def _ecology_completion(subsystem: "TechnologySubsystem") -> float:
    return _track_completion(subsystem, _ECOLOGY_TRACK)


def _medicine_completion(subsystem: "TechnologySubsystem") -> float:
    return _track_completion(subsystem, _MEDICINE_TRACK)


def _genetics_completion(subsystem: "TechnologySubsystem") -> float:
    return _track_completion(subsystem, _GENETICS_TRACK)


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
    :func:`~worldsim.military.ftl.add_ftl_tech_nodes`.
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
    """

    name = "biology"
    _n_inputs = 12

    def _build_state(self, nation: "Nation") -> Optional[List[float]]:
        from ..planets import PLANETS
        planet = PLANETS.get(nation.planet)
        return [
            nation.technology.science / 100.0,               # 0: science (cross-subsystem)
            len(self.unlocked) / max(1, len(self.nodes)),    # 1: biology completion ratio
            len(self.available()) / max(1, len(self.nodes)), # 2: available-node ratio
            nation.resources.get("food", 0.0) / 1000.0,      # 3: food supply pressure
            float(nation.population) / 1_000_000.0,          # 4: population scale
            planet.plague_level if planet else 0.0,          # 5: active plague pressure
            float(len(nation.hospitals)) / 10.0,             # 6: medical infrastructure
            float(len(nation.colonies)) / 10.0,              # 7: colonization scale
            float(nation.star_count) / 10.0,                 # 8: stellar footprint
            _ecology_completion(self),                       # 9: ecology track ratio
            _medicine_completion(self),                      # 10: medicine track ratio
            _genetics_completion(self),                      # 11: genetics track ratio
        ]

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

