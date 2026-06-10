"""The research director: allocates points across the three subsystems."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, TYPE_CHECKING

from .subsystems import (
    BiologySubsystem,
    EngineeringSubsystem,
    PhysicsSubsystem,
    TechnologySubsystem,
)
from .technology import TechnologyNode

if TYPE_CHECKING:
    from ..ai import ResearchAI
    from ..nations.nation import Nation


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
        from ..military.ftl import add_ftl_tech_nodes
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
