"""National projects — data classes, catalog, and nation-level project management."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .nation import Nation


@dataclass(slots=True)
class NationalProject:
    """Large-scale construction tracked at the nation level."""

    name: str
    cost: float
    progress: float = 0.0
    on_complete: Optional[callable] = None
    prereqs: Set[str] = field(default_factory=set)

    def advance(self, amount: float) -> bool:
        """Increase progress by ``amount`` and return ``True`` if finished."""
        remaining = max(self.cost - self.progress, 0.0)
        factor = remaining / self.cost if self.cost else 1.0
        self.progress += amount * factor
        return self.progress >= self.cost


@dataclass(slots=True)
class ProjectSpec:
    """Specification for a buildable national project."""

    cost: float
    on_complete: callable
    prereqs: Set[str] = field(default_factory=set)


PROJECT_CATALOG: Dict[str, ProjectSpec] = {
    "Highway Network": ProjectSpec(
        100.0,
        lambda n: setattr(n, "infrastructure", n.infrastructure + 20),
    ),
    "Research Complex": ProjectSpec(
        80.0,
        lambda n: setattr(
            n.technology, "science", min(100.0, n.technology.science + 10.0)
        ),
        {"Highway Network"},
    ),
    "Orbital Defense Grid": ProjectSpec(
        120.0,
        lambda n: setattr(n, "military", n.military + 20),
    ),
    "Mega Dam": ProjectSpec(
        90.0,
        lambda n: (
            setattr(n, "infrastructure", n.infrastructure + 15),
            setattr(n, "economy_linear", n.economy_linear + 20),
        ),
        {"Highway Network"},
    ),
    "AI Governance System": ProjectSpec(
        110.0,
        lambda n: (
            setattr(n, "stability", min(100.0, n.stability + 20)),
            setattr(n.technology, "industry", min(100.0, n.technology.industry + 10.0)),
        ),
        {"Research Complex"},
    ),
    "Resilience Program": ProjectSpec(
        130.0,
        lambda n: setattr(n, "resilience", min(100.0, n.resilience + 30.0)),
    ),
    "Orbital Shipyard": ProjectSpec(
        150.0,
        lambda n: (
            setattr(n, "military", n.military + 30),
            setattr(n, "infrastructure", n.infrastructure + 10),
        ),
        {"Orbital Defense Grid"},
    ),
}

# Maintains deterministic order for project indexing
PROJECT_NAMES: List[str] = list(PROJECT_CATALOG.keys())


# ---------------------------------------------------------------------------
# Nation-level project management (standalone functions)
# ---------------------------------------------------------------------------

def available_projects(nation: "Nation") -> List[str]:
    """Return project names that can currently be started."""
    opts: List[str] = []
    for pname, spec in PROJECT_CATALOG.items():
        if pname in nation.completed_projects:
            continue
        if any(p.name == pname for p in nation.projects):
            continue
        if not spec.prereqs.issubset(set(nation.completed_projects)):
            continue
        opts.append(pname)
    return opts


def start_project(nation: "Nation", name: Optional[str] = None) -> None:
    """Begin a new national project if resources allow."""
    if len(nation.projects) >= 2 or nation.economy_linear < 50:
        return
    options = available_projects(nation)
    if not options:
        return
    state = nation._civilian_state()
    if name is None or name not in options:
        idx = nation.project_ai.choose_action(state)
        choice = PROJECT_NAMES[idx % len(PROJECT_NAMES)]
        if choice not in options:
            choice = options[0]
    else:
        choice = name
    spec = PROJECT_CATALOG[choice]
    nation.projects.append(
        NationalProject(choice, spec.cost, 0.0, spec.on_complete, spec.prereqs)
    )
    new_state = nation._civilian_state()
    reward = nation.compute_reward("projects", state, new_state)
    nation.project_ai.train(state, PROJECT_NAMES.index(choice), reward, new_state)


def progress_projects(nation: "Nation") -> None:
    """Spend economy to advance national projects."""
    if not nation.projects or nation.economy_linear <= 0:
        return
    invest = min(nation.economy_linear, 20 * len(nation.projects))
    per = invest / len(nation.projects)
    nation.economy_linear -= invest
    finished: List[NationalProject] = []
    for prj in nation.projects:
        if prj.advance(per):
            if prj.on_complete:
                prj.on_complete(nation)
            nation.completed_projects.append(prj.name)
            finished.append(prj)
    for prj in finished:
        if prj in nation.projects:
            nation.projects.remove(prj)
